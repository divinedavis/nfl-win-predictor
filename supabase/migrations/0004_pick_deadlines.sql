-- Lock every pick at kickoff, in the database.
--
-- The page already hides the buttons once a game starts, but that is a
-- courtesy, not a rule: the picks table grants insert/update/delete to
-- authenticated, so a crafted request could still change a losing call after
-- the fact — and a leaderboard that allows that is worth nothing. Postgres
-- needs to know when each game starts, so refresh.sh publishes a deadline for
-- every pickable thing and a trigger refuses writes past it.

create table if not exists public.pick_deadlines (
  season   int not null,
  week     int not null,
  kind     text not null check (kind in ('game', 'prop')),
  ref      text not null,
  kickoff  timestamptz not null,
  primary key (season, week, kind, ref)
);

alter table public.pick_deadlines enable row level security;
-- Written by refresh.sh with the service key; read only through the trigger,
-- which runs as definer. Nothing the browser reaches needs a grant here.
revoke all on public.pick_deadlines from anon, authenticated, public;

create or replace function public.picks_locked()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  row_    record;
  starts  timestamptz;
begin
  row_ := case tg_op when 'DELETE' then old else new end;

  -- Only police end users. auth.uid() is null for a service-role request and
  -- for GoTrue's own account deletion, and without this guard the cascade
  -- from auth.users would trip the lock and roll the whole delete back —
  -- leaving an account that can never be closed once one of its games has
  -- kicked off. The users check covers the same cascade from the other side.
  if auth.uid() is null
     or not exists (select 1 from auth.users u where u.id = row_.user_id) then
    return row_;
  end if;

  select d.kickoff into starts
    from public.pick_deadlines d
   where d.season = row_.season and d.week = row_.week
     and d.kind = row_.kind and d.ref = row_.ref;

  -- No deadline on file means the thing is not gradeable either — no result
  -- will ever join to it — so an unknown ref is allowed through rather than
  -- making every pick impossible if a publish run fails.
  if starts is not null and now() >= starts then
    raise exception 'picks lock at kickoff'
      using errcode = 'check_violation',
            hint = 'This game has already started.';
  end if;
  return row_;
end;
$$;

drop trigger if exists picks_locked_trg on public.picks;
create trigger picks_locked_trg before insert or update or delete on public.picks
  for each row execute function public.picks_locked();
