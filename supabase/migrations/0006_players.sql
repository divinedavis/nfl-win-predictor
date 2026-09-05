-- The list behind "Friends' picks": everyone who has put themselves on the
-- leaderboard, by display name, with the share id their card is reached by.
--
-- share_id was chosen to be unguessable so a card could not be found by
-- walking ids. Listing it here is a deliberate change of meaning: a public
-- profile is now browsable, not just linkable. That is what is_public has
-- promised all along ("the share link shows your picks to anyone who opens
-- it"), and switching is_public off still drops a person from this list and
-- switches their card off in the same stroke. Nothing else is exposed — no
-- user id, no email, no email-derived name.
create or replace function public.players()
returns table (display_name text, share_id text)
language sql
stable
security definer
set search_path = ''
as $$
  select pr.display_name, pr.share_id
  from public.profiles pr
  where pr.is_public
    and exists (select 1 from public.picks p where p.user_id = pr.user_id)
  order by lower(pr.display_name)
  limit 500
$$;

revoke all on function public.players() from public, anon, authenticated;
grant execute on function public.players() to anon, authenticated;
