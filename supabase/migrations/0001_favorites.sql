-- Saved teams and players for signed-in visitors.
--
-- Accounts are OAuth-only (no email/password provider is enabled), and this
-- is the only table the browser touches, so it is written defensively:
-- row-level security scoped to auth.uid(), no grants at all for anon, and a
-- hard per-user row cap so a stolen access token cannot fill the disk.

create table if not exists public.favorites (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  kind        text not null check (kind in ('team', 'player')),
  -- team abbreviation ("KC") or the player's name as it appears in the props
  -- payload; that is the only handle the static page has for a player.
  ref         text not null check (char_length(ref) between 1 and 64),
  label       text check (char_length(label) <= 120),
  created_at  timestamptz not null default now(),
  unique (user_id, kind, ref)
);

create index if not exists favorites_user_idx on public.favorites (user_id, kind);

alter table public.favorites enable row level security;

drop policy if exists favorites_select_own on public.favorites;
create policy favorites_select_own on public.favorites
  for select to authenticated using (user_id = auth.uid());

drop policy if exists favorites_insert_own on public.favorites;
create policy favorites_insert_own on public.favorites
  for insert to authenticated with check (user_id = auth.uid());

drop policy if exists favorites_delete_own on public.favorites;
create policy favorites_delete_own on public.favorites
  for delete to authenticated using (user_id = auth.uid());

-- No updates: a favorite is added or removed, never edited, so there is no
-- update policy and no update grant to go with it.

-- New tables in `public` arrive with ALL privileges already granted to anon
-- and authenticated, so the revoke has to come first or the grant below just
-- layers on top of an existing UPDATE/TRUNCATE.
revoke all on public.favorites from anon, authenticated, public;
grant select, insert, delete on public.favorites to authenticated;

-- Cap the list. 250 is far more than anyone tracks and still bounded.
create or replace function public.favorites_cap()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if (select count(*) from public.favorites where user_id = new.user_id) >= 250 then
    raise exception 'favorite limit reached' using errcode = 'check_violation';
  end if;
  return new;
end;
$$;

drop trigger if exists favorites_cap_trg on public.favorites;
create trigger favorites_cap_trg before insert on public.favorites
  for each row execute function public.favorites_cap();

-- Deleting the account takes the favorites with it (the FK cascades), so the
-- only thing left is the auth.users row, which Supabase owns.
