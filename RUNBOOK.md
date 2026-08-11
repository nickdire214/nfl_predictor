# RUNBOOK

Operational manual for the NFL player-prop prediction system.

> **Note (2026-08-11):** this file did not exist and was created at step 56. Its
> structure is a reconstruction from `DECISIONS.md` and the code, not a restored
> original. Section numbering is new. `DECISIONS.md` remains the authoritative
> record of what was built and why.

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
venv\Scripts\python.exe -m src.models.evaluate           --season YYYY --week N
venv\Scripts\python.exe -m src.models.evaluate_receiving --season YYYY --week N
venv\Scripts\python.exe -m src.models.evaluate_rushing   --season YYYY --week N

REM 2. Refresh raw data
venv\Scripts\python.exe -m src.ingestion.stats
venv\Scripts\python.exe -m src.ingestion.injuries

REM 3. Rebuild feature matrices
venv\Scripts\python.exe -m src.features.engineer
venv\Scripts\python.exe -m src.features.receiving
venv\Scripts\python.exe -m src.features.rushing
```

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

### Wednesday — pull lines, then predict

```
REM 1. Odds snapshot + consensus -> line_overrides
venv\Scripts\python.exe -m src.ingestion.odds

REM 2. Predict the upcoming week (all three markets)
venv\Scripts\python.exe -m src.models.predict           --season YYYY --week N
venv\Scripts\python.exe -m src.models.predict_receiving --season YYYY --week N
venv\Scripts\python.exe -m src.models.predict_rushing   --season YYYY --week N
```

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

Every prediction row carries `as_of` ("`{season} wk{NN}`"), recording the game the
player's roster row was drawn from. 2026 offseason departures are **not
detectable from data alone** — a departed player sits on a 2025-final roster until
the data refreshes. Sort by `as_of` and eyeball the oldest rows.

Rushing and receiving logs also carry `team_implied_total`; rushing carries
`carries_l8`, the prop-slice selector (`carries_l8 >= 8`), so a log can be
filtered to the priced population without rejoining the matrix.

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
| Unresolved prop lines | Book name not matched | `resolve_prop_lines` never guesses — unresolved is by design (a wrong silent match corrupts grading invisibly). Fix by hand. |
| Wide rushing intervals | Intrinsic to the market | Not a defect. See §4. |
| Week-1 Ridge/q50 divergence | Structural, no calibrator history | Expected. See §5.1. |
