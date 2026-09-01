# Review dashboard (read-only)

A local Flask page for the Wednesday review queue. It opens the parquet logs that
the pipeline has already written and displays them.

## Read-only guarantee

**This viewer never writes to `data/`, never triggers a model run, and never
edits an override file.** It lives entirely outside the validated pipeline.

Concretely:

- Every data access is `pandas.read_parquet`, `pandas.read_csv`, or `json.load`.
- No runner (`run_week`, `run_week_receiving`, `run_week_rushing`) is imported.
- No API call is made. Market prices come from prop payloads **already saved** in
  `data/raw/odds/`.
- The one pipeline function it does import is `resolve_starters`, which is itself
  read-only — it reads `qb_matrix.parquet` and `starters_override.csv` and
  returns frames.

If you want to change what the board says, change the override CSVs and re-run
the pipeline. The dashboard will show the result on the next page load.

## Running

```
venv\Scripts\python.exe -m dashboard.app
```

Then open <http://127.0.0.1:5000/>.

Options:

```
--season 2026     default season          --port 5000    listen port
--week 1          default week            --host 127.0.0.1
--dump            print the rendered data to stdout and exit (no server)
```

`--dump` prints exactly what the page would render. Useful for checking output
without a browser.

Other weeks and labeled logs are reachable from the form at the top of the page,
or by URL:

```
/?season=2025&week=8
/?season=2026&week=1&label=props_test
```

## Which file am I looking at?

Per market, the loader prefers the **canonical** log
(`{season}_w{NN}_{market}.parquet`). If there is none, it falls back to the
**most recently modified labeled** file and says so. The banner at the top always
names the exact filename, whether it is canonical or labeled, its row count, and
when it was generated.

Because the fallback is per market, different markets can come from different
runs — the banner is how you notice. Pass `label=` to pin all three to one run.

## Sections

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

## Known limitations (v1)

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
  page load is a couple of seconds.
