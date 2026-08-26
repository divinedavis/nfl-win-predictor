# NFL Predictor

Weekly NFL win probabilities from an XGBoost model blended with an Elo rating
system, trained on free nflverse data (1999–present). Fully local, no API costs.

## Setup

```bash
python3.12 -m venv .venv          # needs Homebrew python3.12+ (system 3.9 is too old)
.venv/bin/pip install -r requirements.txt
# macOS: xgboost needs `brew install libomp`
```

## Usage

```bash
.venv/bin/python features.py      # download fresh data, rebuild features.parquet
.venv/bin/python train.py         # walk-forward backtest + save model.json
.venv/bin/python predict.py       # picks for the next unplayed week
.venv/bin/python predict.py --season 2026 --week 5
.venv/bin/python position_impact.py  # which position's injuries move win prob
.venv/bin/python export_web.py    # rebuild web/index.html for the web app
.venv/bin/python fetch_inactives.py  # ESPN gameday statuses -> inactives.csv
.venv/bin/python fetch_kalshi.py     # Kalshi exchange prices (free, no key)
.venv/bin/python fetch_espn_fpi.py   # ESPN FPI projections (--seasons 2015-2026)
.venv/bin/python fpi_ablation.py     # is FPI worth promoting to a feature?
.venv/bin/python clv.py --record     # lock in today's number vs the market
.venv/bin/python clv.py --report     # closing line value to date
ODDS_API_KEY=... .venv/bin/python fetch_spread_odds.py  # all US books, 2 credits
.venv/bin/python build_qb_splits.py  # per-quarter QB dropback splits (cached CSV)
.venv/bin/python build_qbr.py        # ESPN Total QBR, week + season level
.venv/bin/python qb_grades.py        # the quarter/late-game/playoff report
```

## Individual player ratings

Every QB/RB/WR/TE carries a personal rating: his EPA per game over his last
10 games, shrunk toward zero (`RATING_WINDOW`/`RATING_SHRINK` in features.py),
so rookies and backups sit near 0 and stars ride well above. The model uses:

- **QB ratings** for each game's expected starter (the announced starter
  historically; the incumbent for future games, or 0 if he's ruled out) —
  `qb_val_diff` ranks top-5 in feature importance.
- **Value-weighted outs**: each Out/Doubtful player contributes his rating,
  so Ja'Marr Chase out ≠ a WR4 out. Defenders are valued too — no public
  per-player defensive EPA exists, so DL/LB/DB ratings come from weighted
  box-score playmaking per game (sacks, INTs, passes defended, TFLs, forced
  fumbles; `DEF_VALUE_WEIGHTS`) — Myles Garrett rates ~6.4 vs ~0 for a
  rotational lineman. OL stays snap-weighted (no public stats at all).

`features.py` also snapshots current ratings to `player_ratings.csv`.

## Deployment

Lives at `/opt/nfl-predictor` on the 104.248.12.129 droplet, served by nginx
at `http://104.248.12.129/nfl/` (and `nfl.divinedavis.com` once DNS exists).
`refresh.sh` runs daily at 10:30 UTC via root's crontab: fresh injuries,
weather forecasts, retrain, redeploy to `/var/www/nfl`. `refresh.sh --gameday`
is the fast path — late inactives, fresh market consensus, rebuild, deploy, no
retrain — scheduled in each pre-kickoff window (Sun 11:45 and 15:45 ET, Thu and
Mon 19:15 ET). It takes about a minute; nothing the weekly builds read changes
on a gameday.

`spread_odds_history.csv` accumulates only on the droplet — `/opt/nfl-predictor`
is a plain copy, not a checkout, so the file never flows back to git. Back it up
before rebuilding that box.

## Web app

`export_web.py` bakes the season's predictions into `web/index.html` (from
`web/template.html`) — a self-contained page with a team filter: pick a team,
see the win probability for every game on their schedule, next game featured.
Published as a Claude artifact; republish the same file to update in place.

Re-run all three each week during the season so ratings and rolling form
include the latest results. `predict.py` on a completed week shows HIT/MISS
and the week record.

## How it works

- **Elo** (538-style): K=20, home-field +52, margin-of-victory multiplier,
  1/3 reversion to the mean each offseason, franchise moves handled.
- **Rolling form** (last 8 games, leak-free): point differential, win rate,
  points for/against, offensive EPA, defensive EPA allowed.
- **Injuries** (2009+): players Out/Doubtful and Questionable per team, plus a
  QB-change flag — did the starter change from the team's previous game
  (announced pregame, so leak-free; for future games it fires when the
  incumbent QB is Out/Doubtful on the injury report).
- **Position-weighted outs** (2013+, snap counts): every Out/Doubtful player is
  weighted by his rolling snap share and summed per position group (QB, RB,
  WR, TE, OL, DL, LB, DB) — a full-time starter out ≈ 1.0, a rotational
  backup ≈ his share. `position_impact.py` runs the counterfactual: WR outs
  cost the most (~4% win probability for a full-time WR), then RB; QB shows
  small here only because the QB-change flag already captures it; individual
  defensive absences barely register on single-game outcomes.
- **Weather**: recorded temp/wind for training; free Open-Meteo forecasts at
  predict time for outdoor games within 16 days of kickoff (no API key).
- **Context**: rest days, divisional game, dome, week number.
- **Market**: the no-vig consensus win probability (`mkt_prob`) — the closing
  moneyline with the book's margin removed. See *The market as a feature*.
- **Model**: XGBoost classifier on the above; final probability is
  `0.4 * xgboost + 0.6 * elo` (blend weight swept in backtest — it beats
  either component on calibration).

## Quarterback grades by quarter

`build_qb_splits.py` aggregates every dropback (passes, sacks, scrambles) from
play-by-play into per-quarter buckets — Q1–Q4, the last five minutes of the
fourth or overtime, and that same window with the score inside eight points —
plus a defense-side copy keyed by the defense faced. `build_qbr.py` caches
ESPN Total QBR (week and season level) from the nflverse release.
`qb_grades.py` turns both into the dashboard's **QB grades** tab and prints the
report standalone.

A grade is EPA per dropback inside the bucket, empirical-Bayes shrunk toward
the league mean by that bucket's own noise, then scaled to 50 ± 20 against the
passers shown beside it. Each bucket is scaled on its own, so a 70 in the
fourth quarter means what a 70 in the first does. League-level questions use
raw rates and paired tests, never the graded scale.

Findings (2021–2025, qualified starters):

- **Late-game decline is situational, not personal.** Raw fourth-quarter play
  is +0.025 EPA/dropback against +0.057 in the first three, with completions
  down 61.6% → 58.3% and interceptions up 1.9% → 2.4%. But compare each of 138
  starter-seasons against *himself* in one-score situations and the last five
  minutes come out −0.001 EPA/dropback (p = 0.98). The fourth-quarter dip is
  trailing teams throwing into defenses playing the clock.
- **Playoffs are harder, and about half of it is the opponent.** 66 starters
  with 200+ regular-season and 20+ playoff dropbacks in the same year lose
  −0.089 EPA/dropback in January (37/66 decline, paired t p = 0.007; sack rate
  5.5% → 6.2%). The playoff field allowed −0.045 EPA/dropback more than the
  league in the regular season — roughly half the drop is strength of
  opponent, not the month.
- Both sides of the playoff comparison are shrunk at the playoff run's sample
  size. Grading three games as if they were seventeen manufactures a decline
  out of arithmetic alone.

## Hypothesis scoreboard

Every idea gets the same treatment: build the feature, run the walk-forward
backtest, promote only on a Brier improvement across a clear majority of
seasons. What has been tested so far —

| Hypothesis | Script | Verdict |
|------------|--------|---------|
| Finishing / clutch (late one-score EPA) | `clutch_ablation.py` | **Promoted** — Brier better in 11/11 seasons |
| Kicker quality (distance-adjusted FGOE) | `kicker_ablation.py` | Real skill (split-half r=+0.55), no model gain — Elo absorbs it |
| Coaching + head-to-head matchup history | `coach_ablation.py` | Nothing beyond Elo; residual r=−0.17 |
| Upset indicators (luck, sacks, blitz, travel, letdown) | `backtest_groups.py` | None promoted; two kept as display flags |
| **2+ starting offensive linemen out** | `ol_ablation.py` | Real in raw numbers, already priced — see below |
| Spread from preseason Vegas win totals (RJ's rule) | `wintotals_rule.py` | No edge — 50.3% ATS as stated, 51.3% at best fit; see below |
| **No-vig market probability** | `train.py` | **Promoted** — Brier .2206 → .2164, now the #3 feature |
| Gameday inactives resolving Questionable | `fetch_inactives.py` | Shipped on measured need (63.3% of Q play); no backtest possible — ESPN does not retain past inactives |
| **Early-season ATS edge (weeks 1-4)** | `clv.py` | **Live hypothesis** — 59.1% ATS at edge ≥3 (n=210), monotonic decay, corroborated by the win-totals decay; being measured by CLV, nothing staked |
| ESPN FPI as a feature | `fpi_ablation.py` | **Not promoted** — Brier .2164 → .2162, better in only 6/11 seasons; display only. Corroborates the early-season effect, see below |

The pattern: a feature wins only when it carries information Elo and rolling
EPA could not already have absorbed through game results. The market feature is
the clearest case — it is the one input built from information the model has no
other way to see.

### Offensive line injuries

Teams missing two or more starting linemen score 21.6 points a game and win
44%, versus 23.4 and 52% for teams with a whole line. Nearly all of that gap
is *which* teams get hurt, not the injury — measured against a team's own
last-8-game form, the same teams score +0.16 points (z +0.51) and allow +0.06
percentage points of sack rate (z +0.38). QB, RB and WR production versus each
player's own trailing-4 form moves less than a yard in every direction, and
props pinball loss is flat to worse. `features.py` builds
`{side}_ol_starters_out` and `{side}_ol_multi_out` (the ≥2 threshold) as a
candidate group; neither is in `FEATURES`. The dashboard names a patched-up
line as context, because linemen have no stat line and so never appear in the
key-absence list.

### Preseason win totals as a spread

A rule of thumb passed along by a friend: take each team's preseason Vegas win
total, multiply the gap by 5.5, add 1.75 for home field, and bet that number
against the closing spread — claimed 58% over seven years. `build_win_totals.py`
scrapes the totals (SportsOddsHistory, 2010–2025, plus a hand-entered current
season in `win_totals_current.csv`) and `wintotals_rule.py` grades it.

It does not hold up. As stated it went 50.3% over 4,203 bets (−3.9% ROI) and
49.9% over the last seven seasons. The ×5.5 is the wrong scale: regressing
actual margin on the win-total gap gives **1.76 points per win**, and the
closing line already prices **1.60 points per win** — the market runs this rule
with a better constant. Sweeping the multiplier across its whole range never
clears the 52.4% breakeven (best 51.3%), and no edge filter rescues it.

The one honest hint is early season: at ×1.75, weeks 1–4 hit 54.1% (n=958) and
the edge decays monotonically to 47.9% by week 15 — the right shape for
preseason information going stale as the market learns. It is not significant
(p = 0.155) and it is weaker in the recent half (55.7% in 2010–17 vs 52.5% in
2018–25), so it stays a note, not a bet.

## Against the spread

The dashboard shows an ATS lean next to every game. **It has no measured
edge**: `ats_backtest.py` covers 50.4% over 2,943 games (2015–2025), above the
52.4% breakeven in 3 of 11 seasons. The model's expected margin misses by 10.38
points on average; the closing line misses by 10.08. The line is the better
margin estimate, so a lean built by disagreeing with it loses to the juice.

**Season-wide.** That full-season number is not the last word: split by week,
weeks 1–4 clear breakeven at every edge threshold and weeks 5+ are flat dead —
see *Closing line value* for the evidence and the pre-registered rule now being
measured. The 50.4% here is the average of a real early-season edge and a
slightly negative one for the rest of the year, which is why it reads as
nothing. Nothing is staked on it either way until CLV says so.

An earlier run of this same script recorded 55.2% (10/11 seasons) on
2026-08-13. It does not reproduce under the current features, under the
pre-`ab1cbb3` feature set, or with `features.parquet` rebuilt at the old
`ELO_HFA=52`; a constant-offset sweep peaks at 51.3%, so it was not a
home-field lean either. The number is withdrawn, on the site as well as here.

## Backtest (walk-forward, 2015–2025, 3,018 games)

Each season predicted by a model trained only on prior seasons.

| Predictor            | Accuracy | Brier |
|----------------------|----------|-------|
| Always pick home     | 55.0%    | —     |
| Elo alone            | 64.7%    | .2217 |
| **Blend (shipped)**  | **65.7%**| **.2164** |
| Vegas closing spread | 66.1%    | —     |

Selective picking (`predict.py --min-conf`): confidence is well calibrated,
so acting only on the model's strongest picks buys accuracy —

| Confidence cutoff | Picks/week | Accuracy | Seasons ≥70% |
|-------------------|-----------|----------|--------------|
| ≥ 0.55            | ~10.3     | 68.0%    | 3/11         |
| ≥ 0.58            | ~8.9      | 70.1%    | 5/11         |
| ≥ 0.60            | ~7.9      | 71.6%    | 7/11         |
| ≥ 0.65 (starred)  | ~5.6      | 74.6%    | 10/11        |
| ≥ 0.70            | ~4.0      | 77.8%    | 11/11        |

The last column matters more than the accuracy column: a cutoff that averages
70% while clearing it in 3 seasons of 11 is a number that will embarrass you in
public. Only ≥0.65 and above hold up season by season.

Honest read: single-game NFL outcomes are mostly noise, and ~67% on all games
is the practical ceiling (the Vegas closing line, which prices in everything,
sits at 66.1%). Injury + QB-change + weather features lifted the blend past
plain Elo, and the no-vig market feature closed most of what was left. 70%+ on
every game is not a thing any honest model achieves — 70%+ on the starred
subset is, in 10 seasons out of 11.

`spread_line` is stored in the feature table as a benchmark only; it is never
fed to the model. `mkt_prob` is a different thing and *is* a feature — see
below.

## The market as a feature

The closing line is the best-informed number available before kickoff, so the
model reads it instead of pretending it does not exist. Three pieces:

**No-vig consensus probability (`mkt_prob`).** Both sides of a moneyline imply
probabilities that sum to more than 1; the excess is the book's margin (median
2.5% here). Normalising by the sum removes it and leaves a fair win
probability. nflverse carries closing moneylines for **100% of games
2015–2025**, so this needed no API and no backfill — it was sitting in
`load_schedules()` the whole time. It is now the third most important feature
in the model, behind only the two Elo terms.

Adding it moved the walk-forward numbers from 64.9%/.2206 to **65.7%/.2164**,
and the ≥0.60 bucket from 69.7% to **71.6%**. Feeding the raw `spread_line`
instead reaches a similar accuracy but is less stable season to season — the
spread is a rounded, half-point-quantised summary of the same information, and
the moneyline is the market's actual probability.

**Every US book, for the price of one.** The Odds API bills
`markets × regions`, *not* per bookmaker — verified against the
`x-requests-last` response header, which returns 2 for a two-market,
one-region call regardless of how many books answer. `fetch_spread_odds.py`
used to pin `bookmakers=draftkings`; it now asks for `regions=us` and gets 8
books at the same cost. The consensus is the **median** across books, so one
stale or off-market price cannot drag it, and `spread_odds.csv` now records the
best available price per side rather than DraftKings' price.

**Credit guards.** This call costs 2 credits and now runs five times a week
(daily plus four gameday windows) — 22 a week where it used to be 7. It warns
below 100 remaining, and refuses to run at all with fewer than `CREDIT_RESERVE`
(10) left, because the model falls back to nflverse moneylines when the
consensus file is stale but every other job on the key dies when it hits zero.
The last known balance is cached in `.odds_credits`.

**Line movement is a 2027 feature, not a 2026 one.** Sharp money shows up as
the move from the opening number to the close, and this model only ever sees
the endpoint. There is no free source for the other end: nflverse carries
closing lines only, and ESPN's core-API `open`/`close` blocks come back `null`
for every provider on past seasons (checked on 2023, all 14). Buying the
backfill costs more credits than the plan holds. So `fetch_spread_odds.py` now
**appends** every snapshot to `spread_odds_history.csv` instead of overwriting
— one season of daily snapshots is the cheapest honest path to the feature.
Nothing reads it yet, and nothing should until it covers a full season.

## Gameday inactives

The official injury report is filed Wednesday to Friday. The inactives list
lands 90 minutes before kickoff, and the market moves on it while a model built
on Friday's report is still guessing. Measured against snap counts, 2021–24:

| Report status | Listings | Actually played |
|---------------|----------|-----------------|
| Out           | 4,078    | 0.0%            |
| Doubtful      | 658      | 0.6%            |
| **Questionable** | **6,120** | **63.3%**    |
| not listed    | 12,221   | 88.7%           |

That is **3.2 Questionable players per team per game and a coin flip on each** —
40.6% for quarterbacks, the single most valuable unknown in the feature set.
`QUESTIONABLE_WEIGHT = 0.37` is the right league-wide average and the wrong
answer for any individual game.

`fetch_inactives.py` closes that gap with ESPN's public scoreboard — no API key
— which carries a per-team status block that flips to `Out` when inactives
post. It writes `inactives.csv`; `features.py` reads it and counts a resolved
player in full or not at all instead of at the average. An unresolved
Questionable stays unresolved and keeps the 0.37 fallback, so the file can
never make a game *worse* informed than it was.

Two guards worth knowing about: the fetcher skips anything that is not
`seasontype 2`, because a default August run would otherwise file preseason
week 3 as regular-season week 3; and repeated runs merge by
(season, week, team, player) with the newest status winning, so the Sunday
11:45 run overwrites what Friday recorded.

## What every market thinks

Four opinions sit under each pick on the site — this model, the sportsbook
consensus, Kalshi, and ESPN's FPI — drawn as bars on the same scale, with a
line naming the disagreement when there is one. `sources.py` joins them all to
`game_id`; any that is missing for a game is simply not shown.

**They are not four independent votes.** Ten books is perhaps three or four
real trading desks: the offshore ones largely copy, and `betonlineag` and
`lowvig` are the same operator with different juice. Kalshi is genuinely
separate — a CFTC-regulated exchange where orders match against each other
rather than a house, with **no vig in the price**, so the midpoint is already
a probability. FPI is a separate model built by separate people.

| Source | Independent? | History? |
|---|---|---|
| Sportsbooks (10 via Odds API) | ~3–4 real desks | closing lines only |
| Kalshi | yes | **2 weeks** — settled markets do not go back |
| ESPN FPI | yes | **back to 2015** |

### The Kalshi field-name trap

Kalshi moved its numbers to decimal-string fields. The integer-cent keys
(`yes_bid`, `last_price`, `volume`) still appear in every response and are
**always `None`** — reading them yields an empty dataset with no error of any
kind. Three probes looked like a dead market before `yes_bid_dollars`,
`last_price_dollars`, and `orderbook_fp`/`count_fp` turned up. A book wider
than 15 cents is dropped rather than guessed at: a yawning spread is nobody
making a market, not a cheap edge.

### FPI: tested, not promoted

FPI is the only outside source with real history, so it is the only one that
could be tested rather than merely displayed. It does not clear the bar:

- **Horse race** — adding `fpi_home_prob` to `FEATURES` moves Brier .2164 →
  .2162 and improves in only **6/11 seasons**. The rule in this repo is a clear
  majority, so it stays out.
- **Marginal information over the market** — all games +0.180 [−0.05, +0.46]
  (inconclusive); **weeks 1–4 +0.758 [+0.24, +1.36], 100% of bootstrap draws
  positive**; weeks 5+ +0.013 (nothing).
- **Stacking it with the model and the market makes the bet worse.** A
  leak-free stacked ensemble hits 63.5% in weeks 1–4 against the model's own
  64.9%, and its ATS record on disagreements is 54.0% (n=63) against the
  model's 59.1% (n=210). Better calibration, worse decisions.

### Why it matters anyway

FPI is a completely separate model, and it beats the market **only in weeks
1–4** — exactly where this model does. That is a fourth independent line of
evidence, and it changes what the finding means: the early-season edge is a
property of **the market being soft**, not of this model being lucky. Two
unrelated models both finding information the September line has missed is a
much stronger claim than either one alone.

## Closing line value

Accuracy against the closing line is the one contest the market wins by
construction — the close aggregates everyone's information, this model's
included. `clv.py` scores the question that actually separates a winning bettor
from a lucky one: **when we commit to a number, does the line then move toward
us?** CLV converges in tens of bets rather than hundreds, because it grades the
price instead of the coin flip sitting on top of it, and a model with no edge
averages zero however its season happens to land.

### The pre-registered rule

Fixed in code before the 2026 season, so it cannot be tuned to whatever the
data turns out to show:

> **weeks 1–4, model margin disagrees with the market by ≥ 3.0 points**

Walk-forward 2015–2025, the model's ATS record on its own disagreements:

| Slice | edge ≥ 0 | edge ≥ 3.0 |
|---|---|---|
| **Weeks 1–4** | 53.3% (n=679) | **59.1% (n=210)** |
| Weeks 5+ | 50.6% (n=2,264) | 50.2% (n=677) |

Breakeven is 52.4%. Three things make this worth acting on rather than
dismissing as a slice found by looking:

1. **A regression says the model carries independent information there.**
   Outcomes on market probability *and* a market-blind model probability: the
   model's coefficient is +0.572 [+0.19, +1.03] in weeks 1–4 (99.8% of
   bootstrap draws positive) and +0.029 [−0.20, +0.26] afterwards. Full-season
   it is +0.158 and not significant — the effect only exists early.
2. **The decay is monotonic.** Contiguous four-week windows run 59.1, 55.0,
   52.5, 53.1, 47.9, 45.4, 48.8, 50.5. A lucky window spikes; a real seasonal
   effect slopes.
3. **It reproduces a finding from a different direction.** `wintotals_rule.py`
   independently measured 54.1% in weeks 1–4 decaying to 47.9% by week 15.
4. **There is a mechanism.** In September the market prices off priors because
   no current-season data exists yet.

Robustness: the result holds under both margin maps (59.1% logit, 60.2% under
the repo's linear-in-p map used by `ats_backtest.py`).

**It is not proven.** n=210 at the useful threshold is ~25 bets a season, 7/11
seasons clear breakeven, and the range is 42%–76%. That is precisely why this
ships as a measurement harness and not as a staked bet.

### How it works

`--record` locks in the model's number against the market's the first time a
game has enough books, and **never revises it** — re-recording later would let
a number be chosen with hindsight about which way the line went. `--settle`
attaches the closing number once kickoff has passed, and `--report` splits CLV
by the pre-registered rule versus everything else. Every game is logged, not
just qualifying ones, so the rule is reported against the full field rather
than cherry-picked out of it.

`RULE_MIN_BOOKS = 4` is a data-quality gate, not an outcome filter: weeks 2–4
open months out on two offshore books, and locking a number against that thin a
consensus measures scrape noise rather than the market.

The sign conventions in `clv_points()` and `ats_result()` are covered by
`test_clv.py` — an inverted sign in either would not raise anything, it would
quietly report a losing model as a winning one. (An earlier throwaway version of
the ATS check did exactly that, returning 36% until the cover rule was fixed to
`margin > spread_line`.)

`clv_picks.csv` accumulates **only on the droplet**, like
`spread_odds_history.csv`. `refresh.sh` keeps 14 daily copies in `backups/`, and
`scripts/pull_clv_data.sh` pulls both into the local checkout.

## Player props (paper trading)

Quantile projections (P10/P25/P50/P75/P90) for the four liquid prop markets —
passing yards, rushing yards, receiving yards, receptions — from rolling
usage/efficiency, opponent positional defense, and game context. Walk-forward
validated 2021–2025: 6–10% better pinball loss than a player-history baseline,
with honest coverage (the 50% band catches ~52%, the 80% band ~81–86%).

```bash
.venv/bin/python props.py --validate   # walk-forward backtest
.venv/bin/python props.py              # project next week -> props_projections.csv
ODDS_API_KEY=... .venv/bin/python fetch_props.py   # snapshot DraftKings lines
.venv/bin/python paper_trade.py --pick     # log edges >=5% as $0 paper picks
.venv/bin/python paper_trade.py --settle   # score finished weeks
```

**No real money.** A backtest of the game model against 11 seasons of closing
moneylines lost at every edge threshold (and parlays compound the leak, -10%
ROI), so props follow a pre-committed rule: nothing is staked unless the paper
record clears the 53.3% breakeven (-114 juice) over 200+ settled picks.
Historical prop lines are paywalled — paper trading the live season is the only
honest test. Lines come from The Odds API free tier (hard-capped: one snapshot
per day, one book, four markets).
