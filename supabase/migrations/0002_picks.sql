-- Visitor picks: who wins a game, and whether a player clears his projection.
--
-- Same shape as public.favorites — RLS scoped to auth.uid(), nothing granted
-- to anon, a per-user row cap — with one difference: picks are editable, so
-- unlike favorites this table does grant UPDATE and carries an update policy.
-- Changing a pick is the whole point right up until kickoff.
--
-- Kickoff is enforced in the page, not here: the database has no schedule to
-- check against. Nothing is staked on these, so the only person a late edit
-- fools is the person making it.

create table if not exists public.picks (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  season      int not null check (season between 1999 and 2100),
  week        int not null check (week between 1 and 25),
  kind        text not null check (kind in ('game', 'prop')),
  -- game: "AWAY@HOME". prop: "<player_id>|<stat>".
  ref         text not null check (char_length(ref) between 1 and 80),
  -- game: the team abbreviation picked. prop: 'over' or 'under'.
  choice      text not null check (char_length(choice) between 1 and 8),
  -- prop only: the projection the pick was made against, so a later
  -- re-projection cannot silently move the bar a pick was judged on.
  line        numeric(7, 2),
  -- what the pick was about, in words. The props payload only carries the
  -- current week, so without this a pick from week 3 would render as a bare
  -- player id once week 4 is published.
  label       text check (char_length(label) <= 120),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (user_id, season, week, kind, ref)
);

create index if not exists picks_user_idx on public.picks (user_id, season, week);

alter table public.picks enable row level security;

drop policy if exists picks_select_own on public.picks;
create policy picks_select_own on public.picks
  for select to authenticated using (user_id = auth.uid());

drop policy if exists picks_insert_own on public.picks;
create policy picks_insert_own on public.picks
  for insert to authenticated with check (user_id = auth.uid());

drop policy if exists picks_update_own on public.picks;
create policy picks_update_own on public.picks
  for update to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

drop policy if exists picks_delete_own on public.picks;
create policy picks_delete_own on public.picks
  for delete to authenticated using (user_id = auth.uid());

-- New tables in `public` arrive with ALL privileges already granted to anon
-- and authenticated, so revoke before granting or the grant is a no-op.
revoke all on public.picks from anon, authenticated, public;
grant select, insert, update, delete on public.picks to authenticated;

create or replace function public.picks_touch()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists picks_touch_trg on public.picks;
create trigger picks_touch_trg before update on public.picks
  for each row execute function public.picks_touch();

-- A full season is 272 games plus 48 props a week; 4000 leaves room and still
-- bounds what a stolen access token can write.
create or replace function public.picks_cap()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if (select count(*) from public.picks where user_id = new.user_id) >= 4000 then
    raise exception 'pick limit reached' using errcode = 'check_violation';
  end if;
  return new;
end;
$$;

drop trigger if exists picks_cap_trg on public.picks;
create trigger picks_cap_trg before insert on public.picks
  for each row execute function public.picks_cap();
