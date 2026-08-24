-- Everything the picks page needs beyond the picks themselves: a public
-- identity, the results picks are graded against, and the two read paths that
-- deliberately cross the user boundary (a shared card, and the leaderboard).
--
-- The grading inputs live here rather than in the browser on purpose. A
-- leaderboard scored client-side is a leaderboard anyone can edit with dev
-- tools; refresh.sh publishes results with the service key and Postgres does
-- the arithmetic, so the only thing a visitor can write is their own pick.

-- ---------------------------------------------------------------- profiles
create table if not exists public.profiles (
  user_id      uuid primary key references auth.users (id) on delete cascade,
  display_name text not null check (char_length(display_name) between 1 and 40),
  -- unguessable, so a shared card cannot be found by walking ids. base64url
  -- rather than base64: this ends up in a query string.
  share_id     text not null unique
               default translate(encode(gen_random_bytes(9), 'base64'), '+/', '-_'),
  is_public    boolean not null default true,
  created_at   timestamptz not null default now()
);

-- `create table if not exists` above is a no-op on an existing table, so the
-- column default has to be set on its own or a database created before this
-- edit keeps handing out base64 share ids with "/" and "+" in them.
alter table public.profiles
  alter column share_id set default translate(encode(gen_random_bytes(9), 'base64'), '+/', '-_');
update public.profiles
   set share_id = translate(encode(gen_random_bytes(9), 'base64'), '+/', '-_')
 where share_id ~ '[+/]';

alter table public.profiles enable row level security;

drop policy if exists profiles_select_own on public.profiles;
create policy profiles_select_own on public.profiles
  for select to authenticated using (user_id = auth.uid());

drop policy if exists profiles_update_own on public.profiles;
create policy profiles_update_own on public.profiles
  for update to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

revoke all on public.profiles from anon, authenticated, public;
grant select, update on public.profiles to authenticated;

-- A profile is created for every new account. The default name is a first
-- name and a last initial, not the full name Google hands over: the
-- leaderboard is public and nobody opted into being listed in full.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  raw  text;
  bits text[];
  name text;
begin
  raw := coalesce(new.raw_user_meta_data ->> 'full_name',
                  new.raw_user_meta_data ->> 'name', '');
  bits := regexp_split_to_array(trim(raw), '\s+');
  if raw = '' then
    name := 'Player ' || substr(replace(new.id::text, '-', ''), 1, 4);
  elsif array_length(bits, 1) > 1 then
    name := bits[1] || ' ' || upper(substr(bits[array_length(bits, 1)], 1, 1)) || '.';
  else
    name := bits[1];
  end if;
  insert into public.profiles (user_id, display_name)
  values (new.id, left(name, 40))
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

-- Anyone who signed in before the trigger existed still needs an identity,
-- and re-running this file must stay harmless — hence on conflict do nothing.
insert into public.profiles (user_id, display_name)
select u.id,
       left(case
         when coalesce(u.raw_user_meta_data ->> 'full_name',
                       u.raw_user_meta_data ->> 'name', '') = ''
           then 'Player ' || substr(replace(u.id::text, '-', ''), 1, 4)
         when array_length(regexp_split_to_array(
                trim(coalesce(u.raw_user_meta_data ->> 'full_name',
                              u.raw_user_meta_data ->> 'name')), '\s+'), 1) > 1
           then (regexp_split_to_array(trim(coalesce(u.raw_user_meta_data ->> 'full_name',
                                                     u.raw_user_meta_data ->> 'name')), '\s+'))[1]
                || ' ' || upper(substr((regexp_split_to_array(
                     trim(coalesce(u.raw_user_meta_data ->> 'full_name',
                                   u.raw_user_meta_data ->> 'name')), '\s+'))[
                     array_length(regexp_split_to_array(
                       trim(coalesce(u.raw_user_meta_data ->> 'full_name',
                                     u.raw_user_meta_data ->> 'name')), '\s+'), 1)], 1, 1)) || '.'
         else trim(coalesce(u.raw_user_meta_data ->> 'full_name',
                            u.raw_user_meta_data ->> 'name'))
       end, 40)
from auth.users u
on conflict (user_id) do nothing;

-- ----------------------------------------------------------------- results
-- Written only by refresh.sh with the service key; no role the browser can
-- reach has any privilege here at all.
create table if not exists public.game_results (
  season int not null,
  week   int not null,
  ref    text not null,            -- "AWAY@HOME", matching picks.ref
  winner text not null,
  primary key (season, week, ref)
);

create table if not exists public.prop_results (
  season int not null,
  week   int not null,
  ref    text not null,            -- "<player_id>|<stat>", matching picks.ref
  actual numeric(7, 2) not null,
  primary key (season, week, ref)
);

alter table public.game_results enable row level security;
alter table public.prop_results enable row level security;
revoke all on public.game_results from anon, authenticated, public;
revoke all on public.prop_results from anon, authenticated, public;

-- ------------------------------------------------------------ leaderboard
-- Aggregates only: a display name and a record. No user id, no email, and no
-- way to ask it about one person — which is why anon may call it.
create or replace function public.leaderboard(min_picks int default 3,
                                              limit_n int default 50)
returns table (
  display_name text, wins int, losses int, pct numeric,
  game_wins int, game_losses int, prop_wins int, prop_losses int
)
language sql
stable
security definer
set search_path = ''
as $$
  with graded as (
    select
      p.user_id, p.kind,
      case
        when p.kind = 'game' then
          case when gr.winner is null then null else (p.choice = gr.winner) end
        else
          case
            when pr.actual is null or pr.actual = p.line then null
            else ((p.choice = 'over') = (pr.actual > p.line))
          end
      end as hit
    from public.picks p
    left join public.game_results gr
      on p.kind = 'game' and gr.season = p.season and gr.week = p.week and gr.ref = p.ref
    left join public.prop_results pr
      on p.kind = 'prop' and pr.season = p.season and pr.week = p.week and pr.ref = p.ref
  ),
  tally as (
    select
      g.user_id,
      count(*) filter (where g.hit)::int as wins,
      count(*) filter (where g.hit is false)::int as losses,
      count(*) filter (where g.kind = 'game' and g.hit)::int as game_wins,
      count(*) filter (where g.kind = 'game' and g.hit is false)::int as game_losses,
      count(*) filter (where g.kind = 'prop' and g.hit)::int as prop_wins,
      count(*) filter (where g.kind = 'prop' and g.hit is false)::int as prop_losses
    from graded g
    where g.hit is not null
    group by g.user_id
  )
  select pr.display_name, t.wins, t.losses,
         round(100.0 * t.wins / nullif(t.wins + t.losses, 0), 1) as pct,
         t.game_wins, t.game_losses, t.prop_wins, t.prop_losses
  from tally t
  join public.profiles pr on pr.user_id = t.user_id and pr.is_public
  where t.wins + t.losses >= greatest(min_picks, 1)
  order by pct desc nulls last, (t.wins + t.losses) desc
  limit least(greatest(limit_n, 1), 200)
$$;

revoke all on function public.leaderboard(int, int) from public, anon, authenticated;
grant execute on function public.leaderboard(int, int) to anon, authenticated;

-- ------------------------------------------------------------ shared card
-- Reached only by holding the share_id. Returns a record and the picks
-- themselves — never the owner's id or email.
create or replace function public.shared_card(share text)
returns table (
  display_name text, wins int, losses int, pct numeric, pending int,
  picks jsonb
)
language sql
stable
security definer
set search_path = ''
as $$
  with me as (
    select user_id, display_name from public.profiles
    where share_id = share and is_public
  ),
  graded as (
    select
      p.season, p.week, p.kind, p.ref, p.choice, p.line,
      coalesce(gr.winner, null) as winner,
      pr.actual,
      case
        when p.kind = 'game' then
          case when gr.winner is null then null else (p.choice = gr.winner) end
        else
          case
            when pr.actual is null or pr.actual = p.line then null
            else ((p.choice = 'over') = (pr.actual > p.line))
          end
      end as hit
    from public.picks p
    join me on me.user_id = p.user_id
    left join public.game_results gr
      on p.kind = 'game' and gr.season = p.season and gr.week = p.week and gr.ref = p.ref
    left join public.prop_results pr
      on p.kind = 'prop' and pr.season = p.season and pr.week = p.week and pr.ref = p.ref
  )
  select
    me.display_name,
    (select count(*) filter (where hit) from graded)::int,
    (select count(*) filter (where hit is false) from graded)::int,
    (select round(100.0 * count(*) filter (where hit)
                  / nullif(count(*) filter (where hit is not null), 0), 1) from graded),
    (select count(*) filter (where hit is null) from graded)::int,
    (select coalesce(jsonb_agg(to_jsonb(g) order by g.week desc, g.kind), '[]'::jsonb)
       from (select * from graded limit 200) g)
  from me
$$;

revoke all on function public.shared_card(text) from public, anon, authenticated;
grant execute on function public.shared_card(text) to anon, authenticated;
