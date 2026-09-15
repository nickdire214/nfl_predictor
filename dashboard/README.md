# Dashboard (read-only)

A local Flask app with two pages. It opens parquet files the pipeline has already
written and displays them.

| Page | Route | What it shows |
| --- | --- | --- |
| **Review** | `/` | Forward-looking board for an upcoming week — what we are about to predict. |
| **Evaluation** | `/evaluation` | Backward-looking accuracy, from `*_grades.parquet`. |

Navigate between them with the tabs at the top of either page.

## Read-only guarantee

**This viewer never writes to `data/`, never triggers a model run, never triggers
a grade, and never edits an override file.** It lives entirely outside the
validated pipeline.

Concretely:

- Every data access is `pandas.read_parquet`, `pandas.read_csv`, or `json.load`.
- No runner (`run_week`, `run_week_receiving`, `run_week_rushing`) is imported.
- No API call is made. Market prices come from prop payloads **already saved** in
  `data/raw/odds/`.
- The one pipeline function it does import is `resolve_starters`, which is itself
  read-only — it reads `qb_matrix.parquet` and `starters_override.csv` and
  returns frames.
- The evaluation page imports the three grader modules, but **only for their
  constants** (bucket labels, expected frequencies, `PROP_MIN_TARGETS_L8`,
  `PROP_MIN_CARRIES_L8`). Importing runs no work — each module guards its entry
  point behind `if __name__ == "__main__"`. Sharing the constants is deliberate:
  a second copy of the slice threshold in the dashboard is a number that can
  drift away from the grader's, after which the page would report a population
  the grader never measured.
- The evaluation page **does not grade**. It reads grade files that
  `src.models.evaluate*` already produced. An ungraded week renders an empty
  state telling you to grade it (RUNBOOK §2 step 5), not a grade.

If you want to change what the board says, change the override CSVs and re-run
the pipeline. The dashboard will show the result on the next page load.

## Running

```
venv\Scripts\python.exe -m dashboard.app
```

Then open <http://127.0.0.1:5000/>.

Options:

```
--season 2026       default season        --port 5000    listen port
--week 1            default week          --host 127.0.0.1
--dump              print the rendered REVIEW data and exit (no server)
--dump-evaluation   print the rendered EVALUATION data, both tabs, and exit
```

Both dump flags print exactly what the corresponding page would render. Useful
for checking output without a browser, and for pasting into a write-up.

Other weeks and labeled logs are reachable from the form at the top of the page,
or by URL:

```
/?season=2025&week=8
/?season=2026&week=1&label=props_test

/evaluation?season=2026                        # Last Week tab, latest graded week
/evaluation?season=2026&tab=last_week&week=1   # pick an earlier graded week
/evaluation?season=2026&tab=season             # Season tab (cumulative)
/evaluation?season=2025&tab=season             # a completed replay season
```

## Which file am I looking at?

Per market, the loader prefers the **canonical** log
(`{season}_w{NN}_{market}.parquet`). If there is none, it falls back to the
**most recently modified labeled** file and says so. The banner at the top always
names the exact filename, whether it is canonical or labeled, its row count, and
when it was generated.

Because the fallback is per market, different markets can come from different
runs — the banner is how you notice. Pass `label=` to pin all three to one run.

## Review page — sections

1. **Needs attention** — SKIPped QB teams (blocking), starters whose
   `latest_team` disagrees with their team, lines with no `prob_over`, and
   watchlist players. Every row states why it is flagged.
2. **Prop board** — every priced row across all three markets. Sortable; click a
   column header. Display only: no edge flagging, no highlighting of "good"
   bets, per RUNBOOK §4 (pure-logging mode).
3. **QB starters** — the 32-team table with source, `as_of`, `window_starts`,
   and `latest_team`. SKIP rows are visually distinct.
4. **Roster staleness** — prop-relevant rows sorted oldest `as_of` first, with
   games-back where computable. "Prop-relevant" means `targets_l8 >= 3`
   (receiving) or `carries_l8 >= 8` (rushing); QB has no slice selector, so a
   posted line is the filter there. `slice_basis` names the rule per row.


## Evaluation page — sections

Defaults to the most recent graded week; the dropdown offers any graded week of
the selected season.

### Tab 1 — Last Week

Per market, a card strip: **rows graded**, **point MAE** (prop-slice), **bias**
(subtitled `+ = over-predicted`), **over/under record and hit rate**, and the
roster-quality metric — **starter accuracy** for QB, **availability accuracy**
for receiving and rushing. The exact grade filename, its `generated_at_utc`, and
its file mtime are shown above the cards, because the three markets are graded
independently and can come from different runs.

Then two tables of up to 10 per market: **best** by smallest absolute error,
**worst** by largest. Columns are player, team, predicted, actual, error, line,
`p(over)`, and quantile bucket. Three-way class counts sit beside them.

**Both tables are restricted to the prop-relevant slice** — `actual_targets_l8 >= 3`
for receiving, `actual_carries_l8 >= 8` for rushing, matching the graders' own
thresholds. QB has no slice: all 32 starters are priced. Each header states the
rule and the slice n, e.g. `Best 10 — smallest |error| · prop slice:
actual_targets_l8 >= 3 · n=133`.

The slice is not cosmetic. Ranking every GRADED row by absolute error selects for
**zero-volume players**: being right about someone who was never going to touch
the ball is trivially easy, so the table fills with correct near-zero calls and
tells you nothing. Before this restriction, receiving's best-10 was led by Robert
Tonyan at predicted −0.1 against an actual 0, and seven of its ten rows were
outside the prop population entirely — a correct 74-yard projection cannot compete
with a correct 0. The prop slice is the population the market actually prices, so
it is the population whose best and worst calls are worth reading.

If a market has fewer than 10 rows in its slice, the table shows what exists
rather than padding, and the header's row count reflects that.

**Availability highlighting.** Rows whose miss is an attendance event rather than
a model error are tinted and marked `!`, and the reason is printed beneath. This
is driven off the grader's own `status` column — `DID_NOT_PLAY`, `NO_STATS`,
`WRONG_STARTER` — *not* inferred from a low actual. (The NBA view this is modelled
on tinted rows it guessed were DNPs from low output; that heuristic also flags a
healthy player who simply had a bad game, so it is not reproduced here.)

QB is the one derived case. The QB grader has no class for "started and left" —
the man we named did take the first snap, so the row grades normally and a
120-yard miss looks like model failure. The page flags a starter who was **not
his team's leading passer that week**, measured on pass attempts. That is binary
and workload-based rather than output-based. Validated across four weeks: 2026
wk1 flags exactly Kyler Murray (5 attempts to Carson Wentz's 19) and Sam Darnold
(2 to Drew Lock's 22); 2025 wk1/5/12 flag nobody; 2025 wk18 flags Dak Prescott
and Cam Ward, both rest-or-exit games. No false positives observed.

Two populations are kept apart, because they surface in different places:

- **Flagged** rows *did* grade (a QB who left still has an actual), so they sit
  inside the ranked tables with a marker.
- **Excluded** rows never graded — no actual, no error, so nothing can rank them
  by miss size. They get their own table, ordered by prediction descending, since
  a player projected 60 yards who never took a snap is a larger roster-resolution
  failure than one projected 4. Nothing here counts toward MAE.

### Tab 2 — Season

Cumulative across every graded week of the selected season.

1. **Per-market summary** — weeks graded, total rows, cumulative prop-slice MAE
   and bias, cumulative over/under record, availability.
2. **Week over week**, per market — week, n, MAE, bias, O/U, availability, so
   drift is visible rather than averaged away.
3. **Cumulative quantile buckets**, per market, **prop slice only** — observed vs
   the expected 10/15/25/25/15/10, with weight, deviation, and a ±2 SE band.
   Tie-aware for rushing by *reading the grader's own output*: `evaluate_rushing`
   writes a boundary tie as a pipe-joined label (`q25-q50|q50-q75`), and a tied
   row contributes 1/k to each of its k bands. The tie logic is not reimplemented
   here — a second copy could drift from the grader's.
4. **Over/under by `p(over)` confidence band**, per market and pooled — mean
   predicted probability against the realized rate at which the over actually
   landed, with n per band. This is the direct input to the week 5–6 threshold
   decision.

### How n is displayed, and why it is displayed that way

**Sample size is shown on every table, and the page says so when it is too small
to read.** At one graded week these tables are almost entirely sampling noise,
and a six-cell bucket table rendered without that context invites treating n=32
as if it meant something.

- Banners fire below **300 rows** for bucket tables and **200** for confidence
  bands, stating the actual n against the threshold.
- Each cell carries a **±2 SE** band on the observed proportion, so a deviation
  can be compared against its own noise instead of eyeballed.
- Where a cell is too small for the normal approximation to a binomial
  proportion (fewer than ~5 successes *or* ~5 failures), ±2 SE shows **n/a** and
  the verdict reads **"too few to test"**. An empty cell would otherwise compute
  an SE of exactly 0 and print `±0.0%`, which reads as perfect certainty when it
  is in fact the approximation collapsing.
- The Season tab carries a standing banner at ≤2 graded weeks noting that rows
  inside one week share a single league-wide game environment, so effective n is
  far below the row counts shown.

### No edge, by design

There is no market-probability column, no "our number vs theirs" difference, and
no highlighting of profitable-looking rows anywhere on this page — the same
pure-logging rule the Review board follows (RUNBOOK §4). Until a betting
threshold is set, the purpose is logging, not selection; an edge column shown
before the threshold exists invites acting on it, and one week cannot support
that.

## Known limitations

- **Logs written before step 74 have no `targets_l8`.** Receiving logs gained
  the prop-slice selector at step 74 (matching `carries_l8` in the rushing log).
  Section 4 uses `targets_l8 >= 3` when the column is present and falls back to
  "has a posted line" when it is not; the `slice_basis` column says which rule
  produced each row, so an older log is visibly on the fallback.
- **The watchlist is hard-coded** in `data.py` from RUNBOOK §9. It is hand-curated
  judgement, not anything the pipeline computes, so it needs manual updating.
- **Market prices are matched by normalized player name**, not `gsis_id`, since
  the logs do not carry book prices. A prop with no price match shows `—` rather
  than a guessed number.
- **`latest_team` flags are noise on a historical week.** `latest_team` is a
  *current* roster snapshot, so viewing 2025 wk8 flags every QB who has since
  changed teams (Cooper Rush, Geno Smith, Tua Tagovailoa, Justin Fields on that
  week). That is expected, not a defect — the flag is only meaningful for the
  upcoming week.
- **No caching.** Every request re-reads from disk. Correctness over speed; a
  page load is a couple of seconds. The Season tab re-reads every graded week of
  the season on each request, so a full 22-week season is the slowest view.
- **The receiving model can emit slightly negative predictions.** Three rows in
  2026 wk1 (minimum −1.02 yards, all deep reserves with no posted line). The
  Ridge has no non-negativity constraint. Immaterial to MAE at this magnitude,
  but it is why a prediction column can show a negative number.
- **Availability highlighting for QB is derived, not read from a column.** See
  the Tab 1 notes above. If a quarterback splits snaps by design, the rule would
  flag him; that has not happened in the weeks checked, but it is the failure
  mode to watch.
