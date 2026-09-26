-- Close three holes in the kickoff lock (security audit 2026-09-25).
--
-- 1. A prop pick's `line` came from the browser. Grading compares the actual
--    against picks.line, so a crafted request could save "over 0.5 passing
--    yards" and bank a win. The line now comes from the projection
--    publish_results.py writes to pick_deadlines.line, and whatever the
--    client sent is overwritten. Game picks carry no line at all.
-- 2. The lock only looked at NEW on UPDATE. Rewriting a locked pick's
--    week/ref to a future game moved it past the check, so OLD is now held
--    to the kickoff too, on UPDATE and on DELETE.
-- 3. A ref with no deadline was let through. The whole season's games and
--    every offered prop are published up front, so an unknown ref is either
--    a typo or an attempt to park a pick where no lock can reach it; both
--    are refused now. Choices are also held to the two teams in the ref, or
--    over/under.
--
-- Existing picks are not touched — no UPDATE runs here — so every graded row
-- and the leaderboard stay exactly as they were.

alter table public.pick_deadlines add column if not exists line numeric(7, 2);

create or replace function public.picks_locked()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  owner_  uuid;
  starts  timestamptz;
  line_   numeric(7, 2);
  found_  boolean;
begin
  owner_ := case tg_op when 'DELETE' then old.user_id else new.user_id end;

  -- Only police end users. auth.uid() is null for a service-role request and
  -- for GoTrue's own account deletion, and the cascade from auth.users must
  -- never trip the lock or an account could not be closed once one of its
  -- games kicked off. The users check covers the same cascade from the
  -- other side.
  if auth.uid() is null
     or not exists (select 1 from auth.users u where u.id = owner_) then
    return case tg_op when 'DELETE' then old else new end;
  end if;

  -- The row as it stood: a locked pick can be neither edited nor deleted,
  -- and it cannot escape the lock by being re-pointed at another ref.
  if tg_op in ('UPDATE', 'DELETE') then
    select d.kickoff into starts
      from public.pick_deadlines d
     where d.season = old.season and d.week = old.week
       and d.kind = old.kind and d.ref = old.ref;
    if starts is not null and now() >= starts then
      raise exception 'picks lock at kickoff'
        using errcode = 'check_violation',
              hint = 'This game has already started.';
    end if;
    if tg_op = 'DELETE' then
      -- A pick with no deadline can never be graded; letting its owner
      -- remove it is harmless.
      return old;
    end if;
  end if;

  -- The row as it will be.
  select d.kickoff, d.line, true into starts, line_, found_
    from public.pick_deadlines d
   where d.season = new.season and d.week = new.week
     and d.kind = new.kind and d.ref = new.ref;
  if found_ is not true then
    raise exception 'nothing to pick here'
      using errcode = 'check_violation',
            hint = 'That game or player is not on the board.';
  end if;
  if now() >= starts then
    raise exception 'picks lock at kickoff'
      using errcode = 'check_violation',
            hint = 'This game has already started.';
  end if;

  if new.kind = 'game' then
    if new.choice not in (split_part(new.ref, '@', 1), split_part(new.ref, '@', 2)) then
      raise exception 'pick one of the two teams' using errcode = 'check_violation';
    end if;
    new.line := null;
  else
    if new.choice not in ('over', 'under') then
      raise exception 'pick over or under' using errcode = 'check_violation';
    end if;
    if line_ is null then
      raise exception 'no published line for this player'
        using errcode = 'check_violation';
    end if;
    new.line := line_;
  end if;
  return new;
end;
$$;

-- Trigger functions are never called directly; keep the Data API from
-- listing this one as an RPC.
revoke all on function public.picks_locked() from public, anon, authenticated;

drop trigger if exists picks_locked_trg on public.picks;
create trigger picks_locked_trg before insert or update or delete on public.picks
  for each row execute function public.picks_locked();
