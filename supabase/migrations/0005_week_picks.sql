-- A small "how did everyone else call this?" line under each game.
--
-- Returns the picks of every public profile for one week, with nothing but a
-- display name attached — no user id, no email. The caller's own picks are
-- left out so the page can say "others" and mean it; anon (auth.uid() null)
-- simply sees every public pick. Being public already means a share link
-- shows these picks to anyone who opens it, so this reveals nothing new.
create or replace function public.week_picks(season_ int, week_ int)
returns table (display_name text, kind text, ref text, choice text)
language sql
stable
security definer
set search_path = ''
as $$
  select pr.display_name, p.kind, p.ref, p.choice
  from public.picks p
  join public.profiles pr on pr.user_id = p.user_id and pr.is_public
  where p.season = season_ and p.week = week_
    and p.user_id is distinct from auth.uid()
  order by p.updated_at desc
  limit 2000
$$;

revoke all on function public.week_picks(int, int) from public, anon, authenticated;
grant execute on function public.week_picks(int, int) to anon, authenticated;
