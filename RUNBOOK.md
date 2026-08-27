# RUNBOOK

Operational manual for the NFL player-prop prediction system.

> **Note (2026-08-11):** this file did not exist and was created at step 56. Its
> structure is a reconstruction from `DECISIONS.md` and the code, not a restored
> original. Section numbering is new. `DECISIONS.md` remains the authoritative
> record of what was built and why.
>
> **Amended 2026-08-27 (steps 57–62):** prop-pull mechanics are now real rather
> than forward-looking (§2.1), roster review rewritten after the step-61 roster
> fix (§5), known coverage gaps added (§8), current state added (§9).

Environment: Windows. Run everything through the venv:

```
venv\Scripts\python.exe -m <module> [args]
```

---

## 1. The three markets

| Market | Point estimate | Distribution / `prob_over` | Runner | Grader |
| --- | --- | --- | --- | --- |
| QB passing yards | Ridge + `RollingCalibrator(n0=150)` | pooled empirical residual CDF | `src.models.predict` | `src.models.evaluate` |
| Receiving yards | Ridge + `RollingCalibrator(n0=150)` | position × prediction-conditional sigma + z-pool | `src.models.predict_receiving` | `src.models.evaluate_receiving` |
| RB rushing yards | calibrated Ridge (**headline**) | **static quantile models** (q05…q95) | `src.models.predict_rushing` | `src.models.evaluate_rushing` |

Rushing is **hybrid**: `pred_rushing_yards` (Ridge) and `q50` (quantile model) are
two distinct estimators, deliberately both surfaced and named by source. See §5.

---

## 2. Weekly cadence

### Tuesday — grade last week, then rebuild features

Grade first (graders read the canonical prediction logs, which the rebuild does
not touch), then refresh the raw layer and rebuild the matrices.

```
REM 1. Grade the completed week (all three markets)
REM     PRE-SEASON: skip — there is nothing to grade before week 1 is played.
venv\Scripts\python.exe -m src.models.evaluate           --season YYYY --week N
venv\Scripts\python.exe -m src.models.evaluate_receiving --season YYYY --week N
venv\Scripts\python.exe -m src.models.evaluate_rushing   --season YYYY --week N

REM 2. SNAPSHOT the current matrices BEFORE re-ingesting (see note below)
mkdir data\_feature_backups\YYYY-MM-DD
copy data\features\*.parquet data\_feature_backups\YYYY-MM-DD\

REM 3. Refresh raw data
venv\Scripts\python.exe -m src.ingestion.stats
venv\Scripts\python.exe -m src.ingestion.injuries

REM 4. Rebuild feature matrices
venv\Scripts\python.exe -m src.features.engineer
venv\Scripts\python.exe -m src.features.receiving
venv\Scripts\python.exe -m src.features.rushing
```

**Why the snapshot (step 2) is not optional.** An nflverse refresh can silently
change *historical* values, not just append new ones. After the August 2026
refresh the rushing quantile artifacts moved (crossing rate 5.12% → 5.80% at an
identical row count and identical spine membership) and we could not isolate the
cause: 2025 values were verified unchanged, so the drift originated somewhere in
2021–24 — but no pre-refresh matrix survived to diff against, because `/data/` is
gitignored and the rebuild had already overwritten it. A dated copy costs seconds
and makes that diff possible. Keep at least the last two.

Backups live in `data/_feature_backups/` — a **sibling** of `data/features/`, not
inside it, so nothing that globs the features directory can pick them up.

Proven in the 2026-08-27 dry run: the snapshot diff caught receiving 30,574 →
30,575 (one added row, Jacoby Jones WAS 2025 wk11, from a refreshed crosswalk),
confirmed purely additive with 0 rows lost and QB/rushing unchanged — in seconds,
and with certainty rather than inference.

**Watch the spine builders' output during step 3** — see §6 (label audit).

Artifact rebuilds are **not** part of the weekly cadence. The receiving sigma
table / z-pool and the rushing quantile models are season-static by design; the
point models and calibrators absorb new games each week on their own. Rebuild
artifacts only on a deliberate decision (and log it):

```
venv\Scripts\python.exe -m src.models.train_receiving
venv\Scripts\python.exe scripts\build_receiving_prob_artifacts.py
venv\Scripts\python.exe scripts\build_rushing_prob_artifacts.py
```

### Wednesday — game lines, then props, then predict

Three stages. Game lines feed `line_overrides` (game-script features); player
props feed the runners' `lines=` parameter (`prob_over`). They are separate pulls
with very different costs — see §2.1.

```
REM 1. Game lines: events + spreads/totals (2 credits per snapshot)
venv\Scripts\python.exe -m src.ingestion.odds

REM 2. Player props + predictions — see §2.1 (no CLI; driven from Python)

REM 3. If skipping props, predict without prob_over:
venv\Scripts\python.exe -m src.models.predict           --season YYYY --week N
venv\Scripts\python.exe -m src.models.predict_receiving --season YYYY --week N
venv\Scripts\python.exe -m src.models.predict_rushing   --season YYYY --week N
```

> ⚠ **`-m src.ingestion.odds` is a test harness, not a general weekly command.**
> Its `main()` hardcodes the snapshot label `2026wk1_test_{today}` and hardcodes a
> 2026-week-1 filter in its summary. It fetches events and game lines correctly
> for any week, but the saved filename and the printed summary will be wrong/
> misleading outside 2026 wk1. Either call `fetch_game_lines(snapshot_label)`
> directly with a correct label, or fix `main()` to take `--season/--week`.
> Flagged 2026-08-27; not yet fixed.

---

## 2.1 Prop pull — mechanics

**Endpoint (per-event, not bulk).** Player props are only available on
`/v4/sports/americanfootball_nfl/events/{event_id}/odds`. The bulk `/odds` route
used for game lines does not return them. Wrapper: `fetch_event_props(event_id,
markets, regions="us", save_raw=True)`, which saves the untouched payload to
`data/raw/odds/props_raw_{event_id}_{date}.json`.

Markets: `player_pass_yds,player_reception_yds,player_rush_yds`.

**Cost — 1 credit per market per event, but ONLY for events that return data.**
An event with no props posted costs **0 credits**. So the headline figure is an
upper bound at full coverage, not the expected bill.

| Pull | Cost |
| --- | --- |
| Game lines (whole slate, spreads+totals) | **2 credits** |
| Props, 3 markets × 16 games, **full coverage** | 48 credits (ceiling) |
| Props, 3 markets × 16 games, **measured 2026-08-27** | **9 credits** |

The 2026-08-27 dry run requested all 16 week-1 events and paid 9, because only
3 events had any props posted. Budget against the 48 ceiling, but expect far less
outside the final days before kickoff.

**Coverage ramps as kickoff approaches.** In late August, 3 of 16 week-1 games
had props; the other 13 returned zero bookmakers. Do not read an empty event as
an error — it means the books have not posted yet. Re-pull closer to game day.

**Books are thin and the roster changes.** Props come back from a handful of
books versus 9 for game lines, and some props appear on one book only. Observed
2026-08-26: DraftKings, FanDuel, BetRivers. Observed 2026-08-27: those three plus
**BetOnline.ag**, and the NE@SEA event went from 3 books to 4 overnight. **Do not
hardcode or rely on a fixed book list** — read `n_books` per prop instead. A
median over one book is not a consensus.

**The chain:**

```
fetch_event_props(event_id, markets)          # raw JSON, saved to disk
  -> parse_event_props(raw_json)              # long form: one row per (book, market, player, side)
  -> consensus_prop_lines(props_df)           # one row per (market, player): median line,
                                              #   n_books, over_price/under_price kept SEPARATE
  -> resolve_prop_lines_event(cons, season, week)   # -> gsis_id
  -> runner lines= parameter                  # -> prob_over
```

Over and under prices are medianed **separately** on purpose: a −115/−109 prop is
not the same product as −112/−112, and averaging the sides erases the vig
asymmetry.

Runner `lines=` keys differ — **receiving and rushing key by `gsis_id`; QB keys by
`team`**:

```python
run_week_receiving(season, week, line_overrides=ovr, lines={gsis_id: line}, label="...")
run_week_rushing(  season, week, line_overrides=ovr, lines={gsis_id: line}, label="...")
run_week(          season, week, line_overrides=ovr, lines={team: line},    label="...")
```

**Payload shape — why resolution is event-scoped.** The props endpoint carries
**no per-player team and no player id**. The player name is in the outcome's
`description` field, `name` is `"Over"`/`"Under"`, `point` is the line, `price` is
American odds — and that is the entire outcome object. Team is known only at
event level. (This is inverted from the game-lines endpoint, where `name` carries
the team.)

So `resolve_prop_lines_event` tries **each of the event's two teams separately**
through the unchanged team-scoped `resolve_player`, then arbitrates:

| Outcome | Meaning |
| --- | --- |
| exactly one team matches | resolved; `resolved_team` records which |
| **both** teams match | `ambiguous_both_teams` — two equally good answers |
| neither matches | `unresolved` |

**Always read the resolution report.** The two failure methods mean different
things and need different responses:

- **`unresolved`** — we could not find the player. Usually a rookie (no NFL
  history, so structurally absent from the board) or a roster gap. Check §5.
- **`ambiguous_both_teams`** — two equally good matches, one on each side of the
  game. This needs a **human decision**, not a retry.

**Never force a match.** A wrong silent match corrupts grading invisibly and is
strictly worse than an unresolved line a human can fix.

---

## 3. File and immutability conventions

- Prediction and grade logs live in `data/predictions/` and are **immutable**.
  Runners and graders raise `FileExistsError` unless `force=True` / `--force`.
- Any non-canonical run **must** carry `--label` (e.g. `--label aug_test`), which
  suffixes the filename so it cannot collide with a real weekly log.
- Graders always read the **canonical** (unlabelled) file.
- Never edit a hard-assertion test (`scripts/test_*.py`) to make it pass. The only
  acceptable edit is updating a pinned regression value for a deliberately adopted
  change — and that gets a `DECISIONS.md` row.

---

## 4. Edge thresholds and bet sizing

2026 begins as **paper trading**: every player with a posted line gets a logged
prediction + `prob_over`, with no bet flagging. The edge threshold is deferred
until ~4–6 weeks of live calibration data exist, then **frozen** for the season.

**Per-market threshold guidance:**

- **Rushing needs a more conservative threshold than QB or receiving.** Its
  prop-slice MAE is ~27 yards with honest-but-wide intervals — the intervals are
  trustworthy (2025 replay reproduced nominal within ~2pp on the prop slice), but
  a market this volatile means only **large projection-vs-line gaps** clear a
  useful edge. A gap that would be actionable in the QB market is noise here.
- Receiving and QB prop-slice errors are materially smaller, so a common
  threshold across all three markets would systematically over-bet rushing.

---

## 5. Roster review

**What step 61 automated.** Receiving and rushing rosters are now built from a
**4-game window** (`ROSTER_WINDOW_GAMES = 4` — any player with an offensive snap
for that team across its last 4 games) and then reassigned to
`players.parquet.latest_team`, filtered to `status ∈ {ACT, PUP}`. On the 2026
week-1 board this took stale-team rows from 81 → **0** (receiving) and 23 → **0**
(rushing). Those two markets no longer need a manual team check.

Both changes are **prediction-time only**, gated on
`_latest_team_is_authoritative(players, season)`, which is true only when the
crosswalk describes the season being predicted. A 2025 replay cannot be relabelled
with 2026 teams — verified at 0/288 team mismatches.

Every prediction row still carries `as_of` (the player's **own** most recent game
in the window), so it remains the per-player staleness signal. Rushing and
receiving logs carry `team_implied_total`; rushing carries `carries_l8`, the
prop-slice selector (`carries_l8 >= 8`).

### 5.1 Early-season caveats (weeks 1–4)

Week-1 predictions are structurally different from mid-season ones:

- `calibrator_offset` is **exactly 0** — there are no completed weeks to replay.
- Every `_std` feature is filled from its `_l8` counterpart, and rolling windows
  are thin or empty.

Consequence: the Ridge point estimate and the quantile median **diverge far more
in week 1 than later**, and the Ridge estimate runs hot. 2025 prop slice,
`pred_rushing_yards − q50`:

| Segment | n | mean gap |
| --- | --- | --- |
| week 1 | 21 | **+12.73** |
| weeks 2–18 | 574 | +4.54 |
| playoffs | 35 | +0.28 |

The calibrator is **not** the main driver — its offset averages only +0.33 over
weeks 2–18, ~4% of the week-1 effect. Thin/`_std`-filled features are.

**Which estimator to weight early.** On 2025 graded outcomes, `q50` had lower MAE
than the Ridge headline in *every* segment, and the margin was largest in week 1
(prop slice: Ridge 30.85 vs q50 21.46). Week 1 is also the only segment where
Ridge bias is **positive** (+6.55 prop slice, +8.57 all-graded) — it over-predicts
out of the gate.

Stated at the confidence the evidence supports:

- **Week 1 — provisional (n=21 prop slice):** treat the Ridge headline as running
  hot and lean on `q50` and the quantile intervals. Do not treat the 9.4-yard MAE
  margin as a stable estimate; it rests on 21 rows.
- **Mid-season — better evidenced but partly structural:** `q50`'s ~2–3 yard MAE
  edge holds on n=470–630, but MAE is a *median-optimal* metric and `q50` is a
  median estimate, so this comparison structurally favors it. Ridge is much closer
  to unbiased (full-season prop bias −1.54 vs `q50`'s −6.12). Neither estimator
  dominates: prefer Ridge when you care about bias, `q50` when you care about
  typical error.
- A large Ridge-vs-`q50` divergence remains a **visible sanity signal**, not a
  defect — but in week 1, expect it to be large by default.

### 5.2 What still needs a human — three things

**(a) QB starters are a MANDATORY manual step, because the correct answer is not
in our data.**

This is the important framing, and it took a failed fix to arrive at. Who starts
at quarterback in September is decided by offseason signings, trades, camp
battles, and recoveries — **none of which appear in any input we ingest**. Our
only evidence is who started in 2025. No resolution rule can bridge that gap: a
4-game modal window was tried in step 67 and reverted, because it changed only
2 of 32 teams (one to a QB who is out of the league) and lagged every measured
mid-season change by an extra game. `STARTER_WINDOW_GAMES = 1`, so the default is
simply each team's most recent prior start.

**Treat the default as a starting point for review, never as an answer.** On the
2026 week-1 board, 8 of 32 teams closed 2025 with someone other than their
season-long primary starter — KC (Mahomes out from wk16, Oladokun closing), MIA
(Tua → Ewers from wk16), IND, DEN, LV, ATL, NYJ, WAS — and 3 more had QBs who
have since changed teams (Kirk Cousins → LV, Kenny Pickett → CAR, Josh Johnson →
CIN). Both groups need a human.

**Two review aids**, neither of which is a fix:

- **`latest_team` / `status`** (from `players.parquet`) catches the QB who has
  *moved*. Compare against the resolved team; flag any mismatch, and flag any
  status outside `{ACT, PUP}`. Caught 3 of 32 on the week-1 board.
- **`window_starts`** in the summary shows how contested the window was. At N=1
  it always reads `1/1` — which is itself the reminder that the default rests on
  a single game.

Neither aid sees the "right team, wrong quarterback" case, which is the larger
group: the backup genuinely is on that team, so `latest_team` agrees.

**A populated `starters_override.csv` is the EXPECTED steady state for week 1,
not an exception.** An empty override file in week 1 means the review has not
been done, not that the defaults were correct. Rows matching `(season, week)`
override the default; names resolve against `players.parquet` by exact
`display_name`, and an unknown or ambiguous name raises with candidates listed.

```csv
season,week,team,player_name
2026,1,KC,Patrick Mahomes
2026,1,ATL,SKIP
```

**`SKIP` — when the job is genuinely undecided.** `player_name = SKIP`
(case-insensitive, whitespace-tolerant) drops that team from the board entirely:
it is omitted from the `{team: gsis_id}` dict, but appears in the summary with
`source="skip"` and a null `qb_id`, so a short board is **visible rather than
silent**. The team's opponent is unaffected and still gets predicted normally.

**Use SKIP rather than guessing.** A two-way quarterback competition turns on
front-office decisions — how contract money is committed, whether a veteran is
judged to be declining, whether the organisation is ready to commit to a young
starter — and **none of that is derivable from passing statistics**. Picking the
QB with better 2025 numbers is not analysis, it is a coin flip dressed up as one,
and it produces a logged prediction we would then grade as if it meant something.
Dropping the team costs one row; guessing wrong costs a corrupted record.

The 2026 week-1 file uses SKIP for two teams: **ATL** (Penix Jr. vs Cousins, an
open job) and **LV** (the plausible starter has no NFL history — see below).

**No-history guard.** A named override QB with **zero `qb_matrix` rows** before
`(season, week)` raises a `ValueError` naming his `rookie_season`,
`last_season`, and `latest_team`, and pointing at `SKIP`. Without the guard he
would resolve fine, produce all-NaN rolling features, and be dropped silently in
preprocessing — the add would appear to work and quietly do nothing. This mirrors
the `roster_override.csv` guard (§5.2b). Fernando Mendoza (LV, 2026 rookie) is
the worked example.

So the three override failure modes all raise, none degrade:

| Condition | Result |
| --- | --- |
| Unknown `player_name` | `ValueError` — no QB `display_name` match |
| Ambiguous `player_name` | `ValueError` listing candidate gsis_ids |
| Resolved but no `qb_matrix` history | `ValueError` — points at `SKIP` |

**SKIP is enforced explicitly** (step 70), on future and historical weeks alike.
`build_prediction_features(..., skip_teams=[...])` drops those teams *before* the
schedules fallback, and `run_week` derives the list from the summary
(`source == "skip"`) and passes it through.

This matters because `_build_team_game_view` seeds `qb_id` from schedules'
`home_qb_id`/`away_qb_id`, which is populated for every completed week. Before
the fix, a SKIP row on 2025 wk8 still produced a KC prediction using the
schedule's own Patrick Mahomes, with fully-populated features that survived
preprocessing — the sentinel only appeared to work on future weeks, where those
columns are null and the row happened to die in preprocessing.

> ⚠ **If you call `build_prediction_features` directly, you must pass
> `skip_teams` yourself.** Omitting a team from `starters` alone is not enough —
> it will fall back to the schedule's QB. `run_week` handles this for you.

Mid-season the defaults become reliable — a team's most recent start is a good
predictor of its next one once games are being played — so the override file
should shrink to genuine in-week news (injury, benching) after week 1.

**(b) The 4-game window misses long-absence players.** A player whose last
appearance predates his team's last game by **more than 4 team-games** is absent
from the board entirely — season-ending injuries, mostly. Named 2026 week-1
examples:

| Player | Last game | Team's roster drawn from |
| --- | --- | --- |
| Malik Nabers | 2025 wk4 | NYG wk18 |
| James Conner | 2025 wk3 | ARI wk18 |
| Garrett Wilson | 2025 wk10 | NYJ wk18 |
| Sam LaPorta | 2025 wk10 | DET wk18 |
| Alvin Kamara | 2025 wk12 | NO wk18 |

They will not appear. **If a book posts a line for one of them, that is the signal
to add them manually** — the book knows he is back and we do not. (N=4 recovered
36 of 52 such receivers and 11 of 16 backs; these are the remainder.)

This happened on the first live slate pull: **Malik Nabers came back unresolved
with a posted 60.0 receiving line.**

**The mechanism is `data/predictions/roster_override.csv`** (step 67B), read by
both `build_receiving_prediction_features` and
`build_rushing_prediction_features`:

```csv
season,week,market,team,player_name
2026,1,receiving,NYG,Malik Nabers
```

`market` is `receiving` or `rushing`. The player is injected onto the board with
his **real rolling history** (Nabers arrives with `targets_l8` 9.875,
`receiving_yards_l8` 82.0 from his 2025 weeks 1–4). Rows are scoped to their
exact `(season, week, market)` — a week-1 receiving row does not leak into week 2
or onto the rushing board. If the player is already on the board, the override
wins on team assignment.

Three guards, all raising rather than degrading silently:

| Condition | Result |
| --- | --- |
| Unknown `player_name` | `ValueError` — no `display_name` match |
| Ambiguous `player_name` | `ValueError` listing candidate gsis_ids |
| **Resolved but no spine history** | `ValueError` — every rolling feature would be NaN |

That third guard is why a rookie cannot be forced onto the board: adding Jadarian
Price raises rather than producing a row of NaNs that `preprocess_*` would quietly
drop.

**(c) Rookies are structurally absent, and that is correct.** A player with no NFL
history has all-NaN rolling features and cannot be predicted; the window roster is
derived from the historical spine, so rookies never enter it in the first place
(verified: 0 zero-history players on either final board). **A rookie prop
resolving as `unresolved` is correct behavior, not a bug.** We do not price a
rookie until he has games. Jadarian Price (2026 rookie, SEA) is the worked
example — a real posted line, correctly unresolved.


---

## 6. Snap-label vocabulary audit

Both spine builders (`src.features.rushing`, `src.features.receiving`) print the
full `snap_counts.position` vocabulary among offensive-snap rows and **FLAG any
label outside `KNOWN_SNAP_POSITIONS`**.

**If that FLAG fires during a Tuesday rebuild, stop and investigate before
predicting.** An unrecognized label can silently remove an entire position group
from a spine — this is exactly how the `HB` defect hid: CIN's whole 2025 backfield
was labeled `HB`, which the old exact-match candidacy gate excluded from *both*
the rushing and receiving matrices, with no counter firing. The only visible
symptom was a thin prediction roster three steps downstream.

If the flagged label is a skill-position synonym, add it to `SNAP_LABEL_SYNONYMS`
in `src/features/receiving.py`, rebuild both matrices, and log it in
`DECISIONS.md`.

Healthy output looks like:

```
Snap position labels among 52672 offensive-snap rows:
  WR=13968, TE=8556, RB=7639, ...
  OK: all labels recognized (no unknown synonyms)
```

Candidacy is a **union gate**: a row is admitted if *either* its canonical
position (from `players.parquet`) *or* its synonym-normalized snap label is a
spine position. Canonical position always supplies the position value.

---

## 7. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `FileExistsError` on a runner/grader | Canonical log already written | Intended. Use `--label` for a test run; `--force` only to deliberately overwrite. |
| A team resolves suspiciously few players | Snap-label synonym excluded them, or genuine roster staleness | Check the §6 FLAG line first, then `as_of`. |
| `FLAG: unrecognized snap position label` | New upstream label | Stop. See §6. |
| Unresolved prop line | Book name not matched to the week's pool | By design — never guesses. Usually a rookie (§5.2c) or a long-absence player (§5.2b). Fix by hand. |
| `ambiguous_both_teams` | One name matched a player on each side of the game | Needs a human decision. Do not retry, do not force. §2.1. |
| A prop resolves but gets no `prob_over` | Player not on that market's board | Most often a QB rushing prop — §8. Otherwise §5.2b. |
| `prob_over < 0.5` despite pred > line (receiving) | Right-skewed z-pool | Expected. Crossover is at `line = pred + median(z)·sigma`, i.e. ~0.204·σ **below** pred — a positive margin alone is not enough. Not an inversion. |
| Wide rushing intervals | Intrinsic to the market | Not a defect. See §4. |
| Week-1 Ridge/q50 divergence | Structural, no calibrator history | Expected. See §5.1. |
| Artifacts moved after a refresh, cause unclear | Historical values changed upstream | Diff against the Tuesday snapshot (§2). Without one you cannot isolate it. |

---

## 8. Known coverage gaps

Neither is a defect. Both are places where the books price something we do not.

**QB rushing props.** Books post `player_rush_yds` for quarterbacks on
essentially every game — roughly **2 lines per game, ~32 per slate**. Our rushing
board is **canonical-RB only** by construction (`build_rushing_base` gates on
canonical position `== "RB"`), so these lines **resolve to a valid `gsis_id` but
cannot be priced**. In the week-1 validation this was Drake Maye (23.5) and Sam
Darnold (5.5) — 2 of 3 resolved rushing lines.

**This is a measured decision NOT to build, not an unbuilt backlog item** — see
the 2026-08-27 QB-rushing row in `DECISIONS.md`. The step-64 diagnostic rejected
the two-model design (the distribution is continuous, not bimodal), found naive-L8
already at 12.49 MAE with +0.11 bias, and found 39% of games (pocket QBs) buy only
0.61 MAE over a constant. The binding constraint is a missing feature — there is
no designed-run vs scramble split in `player_stats` — so the real gate is whether
to ingest play-by-play, not how to architect the model. Deferred to the offseason
as a pre-registered experiment against fresh 2026 data. **Do not treat these lines
as a gap to close mid-season.**

**Two-way players.** Travis Hunter is **still canonically CB** in
`players.parquet` after the August 2026 refresh — the refresh did not resolve
this upstream. He sits in the receiving matrix with 7 rows (2025), encodes as
`pos_WR=0, pos_TE=0` (RB baseline), and `sigma_for("CB", x)` returns the **RB**
curve (22.4 at pred=40) rather than the WR curve (32.1). His intervals are
therefore **too tight** for WR-like usage.

**If he has a posted line, apply a manual discount** and treat the interval as
optimistic. The 2025 replay measured him at MAE 28.85 vs a 25.7 WR average with
2× `above_p90` exceedances. A usage-based position override remains deferred.

---

## 9. Current state (2026-08-27)

Status board for Week 1. Sections in parentheses are where the detail lives.

### In place

**Three markets live**, all with runners, graders, immutable logs, and validated
probability layers.

| Market | Roster | Starter/QB |
| --- | --- | --- |
| QB passing yards | n/a | manual review each week (§5.2a) |
| Receiving yards | automated (§5) | n/a |
| RB rushing yards | automated (§5) | n/a |

**Full chain validated end to end against real book payloads**, offline:

| Scope | Consensus pairs | Resolved | Priced |
| --- | --- | --- | --- |
| Single event (NE@SEA) | 14 | 13 | 11 |
| 3-event slate | 31 | 29 | 25 |

Zero fuzzy matches and zero `ambiguous_both_teams` in either run — every
resolution was an exact normalized match.

**QB starters for 2026 wk1 are resolved.** `starters_override.csv` holds 6 named
overrides and 2 SKIPs → **30-team board, zero `latest_team` mismatches** (was 3
before the overrides, plus 5 right-team/wrong-QB cases).

| Team | Starter | Source |
| --- | --- | --- |
| DEN | Bo Nix | override |
| IND | Daniel Jones | override |
| KC | Patrick Mahomes | override |
| MIA | Malik Willis | override |
| NYJ | Geno Smith | override |
| WAS | Jayden Daniels | override |
| ATL | — | **SKIP** |
| LV | — | **SKIP** |

The other 24 teams resolve from the default rule (most recent 2025 start).

**SKIP sentinel and no-history guard both enforced.** Step 70 made SKIP explicit
(`build_prediction_features(..., skip_teams=...)`) rather than incidental — it
previously worked only on future weeks by accident (§5.2a).

**Roster manual-add live** via `roster_override.csv`, currently one row: Malik
Nabers (NYG, receiving) (§5.2b).

**Receiving/rushing pre-season review complete:** 141 receiving + 44 rushing
prop-relevant rows, **zero team mismatches, zero bad statuses**.

**Pure-logging mode.** Every player with a posted line gets a logged prediction
and a `prob_over`. Market probabilities may be displayed alongside ours
(de-vigged), but there is **no edge flagging and no bet recommendation**. The
threshold decision is deferred to **weeks 5–6** and then frozen (§4).

### Open before Week 1, in priority order

1. **ATL quarterback — BLOCKING.** SKIPped until the organization announces.
   Until then there is simply no Atlanta QB prediction. This is the only item
   that blocks a complete board.
2. **Live props pull closer to kickoff.** Coverage was 3/16 events in late
   August and ramps as games approach (§2.1). The full slate has not been pulled
   or resolved yet.
3. **First canonical run.** Everything so far has been labeled. Week 1 is the
   first unlabeled, immutable write — no `--label`, no `--force`.

Lower priority, not blocking: `-m src.ingestion.odds` `main()` hardcodes a
2026-wk1 label/filter (§2); no CLI wraps the prop pull (§2.1); unexplained small
artifact drift from the August refresh, now guarded by the Tuesday snapshot (§2).

### Manual-discount watchlist

These will look like every other row on the board but rest on far less. Discount
them by hand; nothing in the pipeline marks them.

| Player | Issue |
| --- | --- |
| Jawhar Jordan (HOU, RB) | 4 games of history, `carries_l8` 10.75 — inside the prop slice |
| Theo Wease (MIA, WR) | 3 games, sitting at the slice boundary |
| Malik Willis (MIA, QB) | 6 career games, 1 from 2025 — thinnest history on the QB board |
| Malik Nabers (NYG, WR) | form from four September games, 13 team-games stale; live 60.0 line |
| Travis Hunter | still canonically CB → RB-baseline σ despite WR usage (§8) |

### Known non-problems

Do not re-investigate these; each is measured and settled.

- **~4 QB-rushing lines per game resolve but cannot be priced** — a measured
  decision not to build, not a gap (§8).
- **Rookies resolving as `unresolved`** — correct behavior. No NFL history means
  no rolling features (§5.2b).
- **Players at 2.375 targets / 3.75 carries sitting outside the prop slice while
  books price them** — the threshold working as designed, not a roster miss.
