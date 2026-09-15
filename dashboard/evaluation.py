"""Data loading for the read-only EVALUATION page.

READ-ONLY BY CONSTRUCTION, same guarantee as dashboard/data.py: every function
here opens a parquet with pandas.read_parquet and returns frames or dicts.
Nothing writes, nothing triggers a model run, and nothing triggers a GRADE --
this module reads `*_grades.parquet` files that `src.models.evaluate*` already
produced. If a week has not been graded, the page says so; it does not grade it.

The grader modules ARE imported, but only for their CONSTANTS (bucket labels,
expected frequencies, prop-slice thresholds). Importing them executes no work --
every module guards its entry point behind `if __name__ == "__main__"`. Reusing
the constants is deliberate: a second copy of `PROP_MIN_TARGETS_L8` in the
dashboard is a number that can silently drift away from the grader's, and then
the page would report a slice the grader never measured.

WHY THIS PAGE DELIBERATELY DOES NOT SHOW EDGE. There is no market-probability
column, no "our number vs theirs" difference, and no highlighting of rows that
look profitable. Until a betting threshold is set (planned for weeks 5-6), the
purpose of this page is logging, not selection. Presenting an edge column before
the threshold exists would invite acting on it, and one week of rows cannot
support that. The review page already carries market prices for the forward-
looking board; this page is strictly backward-looking accuracy.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import EXPECTED_BUCKET_FREQ, QUANTILE_BUCKETS
from src.models.evaluate_receiving import PROP_MIN_TARGETS_L8
from src.models.evaluate_rushing import (
    PROP_MIN_CARRIES_L8,
    RUSHING_EXPECTED_BUCKET_FREQ,
    RUSHING_QUANTILE_BUCKETS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Per-market grading geometry. `slice_col` is the ACTUAL-side column the grader
# slices on (actual_targets_l8, not targets_l8) -- the prop population is defined
# by what the player actually did, which is what evaluate_receiving.py line 145
# and evaluate_rushing.py line 172 use. Getting this wrong would quietly report a
# different population than the grader's printed summary.
EVAL_MARKETS = {
    "qb": {
        "stem": "qb_pass_yards",
        "label": "QB passing",
        "pred_col": "pred_passing_yards",
        "actual_col": "actual_passing_yards",
        "name_col": "qb_name",
        "id_col": "qb_id",
        "slice_col": None,
        "slice_min": None,
        "slice_note": "no prop slice — all 32 starters are priced",
        "buckets": QUANTILE_BUCKETS,
        "expected": EXPECTED_BUCKET_FREQ,
        "accuracy_label": "starter accuracy",
        # QB's third class is WRONG_STARTER, not DID_NOT_PLAY: the QB board is
        # one row per team, so the failure mode is naming the wrong man rather
        # than rostering someone inactive.
        "miss_class": "WRONG_STARTER",
        "point_label": "point MAE",
    },
    "receiving": {
        "stem": "receiving_yards",
        "label": "Receiving",
        "pred_col": "pred_receiving_yards",
        "actual_col": "actual_receiving_yards",
        "name_col": "player",
        "id_col": "gsis_id",
        "slice_col": "actual_targets_l8",
        "slice_min": PROP_MIN_TARGETS_L8,
        "slice_note": f"actual_targets_l8 >= {PROP_MIN_TARGETS_L8}",
        "buckets": QUANTILE_BUCKETS,
        "expected": EXPECTED_BUCKET_FREQ,
        "accuracy_label": "availability accuracy",
        "miss_class": "DID_NOT_PLAY",
        "point_label": "point MAE",
    },
    "rushing": {
        "stem": "rushing_yards",
        "label": "RB rushing",
        "pred_col": "pred_rushing_yards",
        "actual_col": "actual_rushing_yards",
        "name_col": "player",
        "id_col": "gsis_id",
        "slice_col": "actual_carries_l8",
        "slice_min": PROP_MIN_CARRIES_L8,
        "slice_note": f"actual_carries_l8 >= {PROP_MIN_CARRIES_L8}",
        "buckets": RUSHING_QUANTILE_BUCKETS,
        "expected": RUSHING_EXPECTED_BUCKET_FREQ,
        "accuracy_label": "availability accuracy",
        "miss_class": "DID_NOT_PLAY",
        # Graded on the calibrated-Ridge headline, not q50 -- the two are
        # different estimators (RUNBOOK §1) and only the former is comparable
        # to the other markets' MAE.
        "point_label": "point MAE (calibrated-Ridge)",
    },
}

# Below this many rows, a six-cell bucket table is dominated by sampling noise
# and the page says so instead of inviting a read. At n=133 (receiving week 1,
# prop slice) the ±2 SE band on a 25% cell is roughly ±7.5 points, which is
# wider than most deviations anyone would want to act on.
LOW_N_BUCKETS = 300
LOW_N_CONFIDENCE = 200
MIN_CELL_N = 20

CONFIDENCE_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ------------------------------------------------------------------ discovery

def graded_weeks(season):
    """Weeks with at least one *_grades.parquet, ascending. Never raises."""
    weeks = set()
    for p in PREDICTIONS_DIR.glob(f"{season}_w*_grades.parquet"):
        m = re.match(rf"{season}_w(\d{{2}})_.+_grades", p.stem)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)


def latest_graded_week(season):
    weeks = graded_weeks(season)
    return weeks[-1] if weeks else None


def load_grade_file(season, week, market):
    """Load one market's grades. Returns df + provenance, never raises."""
    cfg = EVAL_MARKETS[market]
    path = PREDICTIONS_DIR / f"{season}_w{week:02d}_{cfg['stem']}_grades.parquet"
    out = {
        "market": market,
        "market_label": cfg["label"],
        "filename": path.name,
        "exists": path.exists(),
        "generated": None,
        "mtime": None,
        "df": pd.DataFrame(),
        "error": None,
    }
    if not path.exists():
        out["error"] = "not graded"
        return out
    try:
        out["df"] = pd.read_parquet(path)
        out["mtime"] = pd.Timestamp(path.stat().st_mtime, unit="s").strftime(
            "%Y-%m-%d %H:%M:%S")
        if "generated_at_utc" in out["df"].columns and not out["df"].empty:
            out["generated"] = str(out["df"]["generated_at_utc"].max())[:19]
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_all_grades(season, week):
    return {m: load_grade_file(season, week, m) for m in EVAL_MARKETS}


# ------------------------------------------------- availability-event flagging

def qb_early_exit_ids(season, week):
    """QB ids who started but were NOT their team's leading passer that week.

    This is the QB analogue of DID_NOT_PLAY, and it is DERIVED rather than read
    from a status column -- the QB grader has no class for "started and left",
    because the man we named did take the first snap, so the row grades as
    GRADED and the huge error looks like model failure.

    The rule is binary and workload-based, deliberately NOT a guess from low
    yardage (the NBA view's red-for-likely-DNP heuristic infers absence from
    output, which would also flag a healthy quarterback who simply played
    badly). A quarterback who takes his team's snaps IS its leading passer; one
    who is out-thrown by a teammate left the game, was benched, or was rested.

    Validated on four weeks: 2026 wk1 flags exactly Kyler Murray (5 attempts to
    Carson Wentz's 19) and Sam Darnold (2 to Drew Lock's 22); 2025 wk1/5/12 flag
    nobody; 2025 wk18 flags Dak Prescott and Cam Ward, both rest-or-exit games
    carrying 187 and 141 yards of error. No false positives observed.

    Returns a dict {qb_id: reason}. Empty on any failure -- a missing raw file
    must degrade the highlight, not the page.
    """
    try:
        ps = pd.read_parquet(RAW_DIR / "player_stats.parquet",
                             columns=["season", "week", "team", "player_id",
                                      "player_display_name", "attempts"])
    except Exception:                                           # noqa: BLE001
        return {}

    wk = ps[(ps["season"] == season) & (ps["week"] == week) & (ps["attempts"] > 0)]
    if wk.empty:
        return {}

    leaders = (wk.sort_values("attempts", ascending=False)
                 .groupby("team").head(1)
                 .set_index("team"))

    out = {}
    for _, row in wk.iterrows():
        lead = leaders.loc[row["team"]]
        if row["player_id"] != lead["player_id"]:
            out[row["player_id"]] = (
                f"{int(row['attempts'])} att vs {lead['player_display_name']}'s "
                f"{int(lead['attempts'])} — did not finish as team's passer")
    return out


def _availability_flag(row, cfg, early_exits):
    """(is_availability_event, reason) for one graded row."""
    status = row.get("status")
    if status == "DID_NOT_PLAY":
        return True, "classified DID_NOT_PLAY — rostered but inactive"
    if status == "NO_STATS":
        return True, "classified NO_STATS — took no offensive snap"
    if status == "WRONG_STARTER":
        return True, "classified WRONG_STARTER — a different QB started"
    pid = row.get(cfg["id_col"])
    if pid in early_exits:
        return True, early_exits[pid]
    return False, None


# ----------------------------------------------------------------- statistics

def _slice_mask(df, cfg):
    """The prop-relevant slice, matching the grader's definition exactly."""
    graded = df["status"] == "GRADED"
    col = cfg["slice_col"]
    if col is None or col not in df.columns:
        return graded
    return graded & (df[col] >= cfg["slice_min"])


def _ou_record(df):
    """(wins, n, rate) over rows carrying an over/under verdict."""
    if "over_under_correct" not in df.columns:
        return 0, 0, None
    s = df["over_under_correct"].dropna()
    if s.empty:
        return 0, 0, None
    wins = int(s.astype(bool).sum())
    return wins, int(len(s)), wins / len(s)


def market_summary(season, week, market, grades):
    """Card-strip metrics for one market in one week."""
    cfg = EVAL_MARKETS[market]
    df = grades["df"]
    out = {
        "market": market,
        "label": cfg["label"],
        "filename": grades["filename"],
        "generated": grades["generated"],
        "mtime": grades["mtime"],
        "exists": grades["exists"],
        "error": grades["error"],
        "slice_note": cfg["slice_note"],
        "has_slice": cfg["slice_col"] is not None,
        "point_label": cfg["point_label"],
        "accuracy_label": cfg["accuracy_label"],
        "classes": {},
        "n_rows": 0, "n_graded": 0,
        "mae": None, "bias": None, "n_slice": None,
        "mae_all": None, "bias_all": None,
        "accuracy": None, "accuracy_num": None, "accuracy_den": None,
        "ou_wins": 0, "ou_n": 0, "ou_rate": None,
        "n_availability": 0, "n_availability_excluded": 0,
        "n_availability_flagged": 0,
    }
    if df.empty:
        return out

    out["n_rows"] = len(df)
    out["classes"] = {k: int(v) for k, v in df["status"].value_counts().items()}
    graded = df[df["status"] == "GRADED"]
    out["n_graded"] = len(graded)

    if not graded.empty:
        out["mae_all"] = float(graded["abs_error"].mean())
        out["bias_all"] = float(graded["error"].mean())

    sl = df[_slice_mask(df, cfg)]
    out["n_slice"] = len(sl)
    if not sl.empty:
        out["mae"] = float(sl["abs_error"].mean())
        out["bias"] = float(sl["error"].mean())

    # Accuracy denominators mirror the graders exactly: QB excludes NO_STATS
    # (a data gap is not a starter miss); receiving/rushing likewise.
    n_graded = out["classes"].get("GRADED", 0)
    n_miss = out["classes"].get(cfg["miss_class"], 0)
    den = n_graded + n_miss
    if den:
        out["accuracy_num"], out["accuracy_den"] = n_graded, den
        out["accuracy"] = n_graded / den

    out["ou_wins"], out["ou_n"], out["ou_rate"] = _ou_record(df)

    # Two distinct populations, kept apart because they surface in different
    # places. EXCLUDED rows never reached grading (no actual, no error) and are
    # listed in their own table. FLAGGED rows DID grade -- a QB who started and
    # left still has an actual -- so they sit inside the ranked tables carrying
    # a marker. Reporting one total for both made the QB section read
    # "0 shown of 2" when both of its events were flagged, not excluded.
    early = qb_early_exit_ids(season, week) if market == "qb" else {}
    miss_classes = {"DID_NOT_PLAY", "NO_STATS", "WRONG_STARTER"}
    out["n_availability_excluded"] = int(df["status"].isin(miss_classes).sum())
    out["n_availability_flagged"] = sum(
        1 for _, r in df.iterrows()
        if r.get("status") == "GRADED" and _availability_flag(r, cfg, early)[0])
    out["n_availability"] = (out["n_availability_excluded"]
                             + out["n_availability_flagged"])
    return out


def best_worst(season, week, market, grades, n=10):
    """(best, worst) lists of n rows by absolute error, PROP SLICE ONLY.

    Ranked over the prop-relevant slice rather than all GRADED rows (step 90).
    Ranking the whole board by absolute error selects for zero-volume players:
    a deep reserve predicted -0.1 who catches nothing scores a better "best"
    than a correct 74-yard call, because being right about a player who was
    never going to touch the ball is trivially easy. Receiving's best-10 was
    led by exactly that row. The slice (actual_targets_l8 >= 3 receiving,
    actual_carries_l8 >= 8 rushing, matching the graders) is the population the
    prop market actually prices, so it is the population whose best and worst
    calls are worth looking at. QB is unchanged -- all 32 starters are priced,
    so it has no slice.

    Still GRADED-only within that slice: an ungraded row has no error to rank.
    The availability flag therefore marks a graded row whose miss is an
    attendance event (a QB who started and left); rows that never graded are
    reported separately by `availability_rows`.

    Returns fewer than n rows when the slice holds fewer -- no padding.
    """
    cfg = EVAL_MARKETS[market]
    df = grades["df"]
    if df.empty or "abs_error" not in df.columns:
        return [], []

    graded = df[_slice_mask(df, cfg) & df["abs_error"].notna()].copy()
    if graded.empty:
        return [], []

    early = qb_early_exit_ids(season, week) if market == "qb" else {}

    def rows(frame):
        out = []
        for _, r in frame.iterrows():
            flag, reason = _availability_flag(r, cfg, early)
            out.append({
                "player": r.get(cfg["name_col"]),
                "team": r.get("team"),
                "position": r.get("position"),
                "pred": _f(r.get(cfg["pred_col"])),
                "actual": _f(r.get(cfg["actual_col"])),
                "error": _f(r.get("error")),
                "abs_error": _f(r.get("abs_error")),
                "line": _f(r.get("line")),
                "prob_over": _f(r.get("prob_over")),
                "bucket": r.get("quantile_bucket"),
                "status": r.get("status"),
                "ou": (None if pd.isna(r.get("over_under_correct"))
                       else bool(r.get("over_under_correct"))),
                "availability": flag,
                "availability_reason": reason,
            })
        return out

    best = rows(graded.nsmallest(n, "abs_error"))
    worst = rows(graded.nlargest(n, "abs_error"))
    return best, worst


def availability_rows(season, week, market, grades, n=10):
    """Board rows excluded from grading because the player did not play.

    These CANNOT appear in the best/worst tables: an ungraded row has no actual
    and no error, so there is nothing to rank it by. They are still misses in
    the operational sense -- we put the player on the board and priced him --
    so they are surfaced separately, ordered by prediction, largest first: a
    player we projected 60 yards for who never took a snap is a bigger
    roster-resolution failure than one we projected 4 for.

    Deliberately kept apart from model error. Nothing here counts toward MAE.
    """
    cfg = EVAL_MARKETS[market]
    df = grades["df"]
    if df.empty or "status" not in df.columns:
        return []
    miss = df[df["status"].isin(["DID_NOT_PLAY", "NO_STATS", "WRONG_STARTER"])]
    if miss.empty:
        return []
    out = []
    for _, r in miss.nlargest(n, cfg["pred_col"]).iterrows():
        _, reason = _availability_flag(r, cfg, {})
        out.append({
            "player": r.get(cfg["name_col"]),
            "team": r.get("team"),
            "position": r.get("position"),
            "pred": _f(r.get(cfg["pred_col"])),
            "line": _f(r.get("line")),
            "prob_over": _f(r.get("prob_over")),
            "status": r.get("status"),
            "reason": reason,
        })
    return out


def _f(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


# -------------------------------------------------------------- season tables

def load_season_grades(season, market):
    """Concatenate every graded week for one market. Returns (df, weeks)."""
    frames, weeks = [], []
    for wk in graded_weeks(season):
        g = load_grade_file(season, wk, market)
        if g["exists"] and not g["df"].empty and g["error"] is None:
            frames.append(g["df"])
            weeks.append(wk)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), weeks


def season_summary(season, market, df, weeks):
    cfg = EVAL_MARKETS[market]
    out = {
        "market": market, "label": cfg["label"],
        "n_weeks": len(weeks), "weeks": weeks,
        "n_rows": len(df), "n_graded": 0, "n_slice": 0,
        "mae": None, "bias": None, "mae_all": None, "bias_all": None,
        "ou_wins": 0, "ou_n": 0, "ou_rate": None,
        "accuracy": None, "accuracy_num": None, "accuracy_den": None,
        "slice_note": cfg["slice_note"], "point_label": cfg["point_label"],
        "has_slice": cfg["slice_col"] is not None,
        "accuracy_label": cfg["accuracy_label"],
    }
    if df.empty:
        return out
    graded = df[df["status"] == "GRADED"]
    out["n_graded"] = len(graded)
    if not graded.empty:
        out["mae_all"] = float(graded["abs_error"].mean())
        out["bias_all"] = float(graded["error"].mean())
    sl = df[_slice_mask(df, cfg)]
    out["n_slice"] = len(sl)
    if not sl.empty:
        out["mae"] = float(sl["abs_error"].mean())
        out["bias"] = float(sl["error"].mean())
    out["ou_wins"], out["ou_n"], out["ou_rate"] = _ou_record(df)
    counts = df["status"].value_counts().to_dict()
    den = int(counts.get("GRADED", 0)) + int(counts.get(cfg["miss_class"], 0))
    if den:
        out["accuracy_num"] = int(counts.get("GRADED", 0))
        out["accuracy_den"] = den
        out["accuracy"] = out["accuracy_num"] / den
    return out


def week_over_week(season, market):
    """One row per graded week so drift is visible rather than averaged away."""
    cfg = EVAL_MARKETS[market]
    rows = []
    for wk in graded_weeks(season):
        g = load_grade_file(season, wk, market)
        if not g["exists"] or g["df"].empty or g["error"]:
            continue
        df = g["df"]
        sl = df[_slice_mask(df, cfg)]
        wins, n_ou, rate = _ou_record(df)
        counts = df["status"].value_counts().to_dict()
        den = int(counts.get("GRADED", 0)) + int(counts.get(cfg["miss_class"], 0))
        rows.append({
            "week": wk,
            "n_rows": len(df),
            "n_graded": int(counts.get("GRADED", 0)),
            "n_slice": len(sl),
            "mae": float(sl["abs_error"].mean()) if not sl.empty else None,
            "bias": float(sl["error"].mean()) if not sl.empty else None,
            "ou_wins": wins, "ou_n": n_ou, "ou_rate": rate,
            "accuracy": (int(counts.get("GRADED", 0)) / den) if den else None,
            "accuracy_num": int(counts.get("GRADED", 0)),
            "accuracy_den": den,
        })
    return rows


def cumulative_buckets(season, market, df):
    """Cumulative quantile-bucket table, PROP SLICE ONLY, observed vs expected.

    Tie-aware by READING the grader's persisted output rather than recomputing
    from the quantile columns: `evaluate_rushing` writes a boundary tie as a
    pipe-joined label ("q25-q50|q50-q75"), so a tied row contributes 1/k to each
    of its k bands. Recomputing here would duplicate the grader's tie logic and
    risk the two drifting apart.

    Reports ±2 SE on each observed share, because a six-cell table over a few
    hundred rows has bands wide enough to swallow most deviations, and a bare
    observed-vs-expected column invites reading noise as signal.
    """
    cfg = EVAL_MARKETS[market]
    buckets, expected = cfg["buckets"], cfg["expected"]
    weights = {b: 0.0 for b in buckets}
    total = 0.0
    n_ties = 0

    if not df.empty:
        sl = df[_slice_mask(df, cfg)]
        for raw in sl["quantile_bucket"].dropna():
            parts = [p for p in str(raw).split("|") if p in weights]
            if not parts:
                continue
            if len(parts) > 1:
                n_ties += 1
            for p in parts:
                weights[p] += 1.0 / len(parts)
            total += 1.0

    rows = []
    for b in buckets:
        w = weights[b]
        obs = (w / total) if total else None
        exp = expected[b]
        # The normal approximation to a binomial proportion needs roughly 5
        # successes AND 5 failures. An empty cell yields se == 0, which would
        # otherwise print as "+/-0.0%" and read as perfect certainty when it is
        # in fact the approximation breaking down. Report n/a instead.
        valid = bool(total) and min(w, total - w) >= 5
        se = math.sqrt(obs * (1 - obs) / total) if (valid and obs is not None) else None
        rows.append({
            "bucket": b,
            "weight": round(w, 1),
            "observed": obs,
            "expected": exp,
            "deviation": (obs - exp) if obs is not None else None,
            "se2": (2 * se) if se is not None else None,
            "se_valid": valid,
            "outside_noise": (bool(abs(obs - exp) > 2 * se)
                              if (obs is not None and se) else False),
            "thin": total > 0 and w < MIN_CELL_N,
        })
    return {
        "rows": rows, "n": int(round(total)), "n_ties": n_ties,
        "low_n": total < LOW_N_BUCKETS,
        "low_n_threshold": LOW_N_BUCKETS,
        "slice_note": cfg["slice_note"],
    }


def confidence_buckets(season, market, df):
    """Realized hit rate by our own prob_over bucket — the threshold input.

    For each band of prob_over, the mean predicted probability against the
    realized rate at which the OVER actually landed. If the model's
    probabilities are honest these track; systematic divergence in a band is
    what a betting threshold would eventually key on.

    NOTE: an over/under verdict is scored against the posted line, so this is a
    calibration check on rows that carried a price, not on the whole board.
    """
    rows = []
    if df.empty or "prob_over" not in df.columns:
        return {"rows": rows, "n": 0, "low_n": True,
                "low_n_threshold": LOW_N_CONFIDENCE}

    d = df[(df["status"] == "GRADED") & df["prob_over"].notna()
           & df["line"].notna() & df[EVAL_MARKETS[market]["actual_col"]].notna()].copy()
    if d.empty:
        return {"rows": rows, "n": 0, "low_n": True,
                "low_n_threshold": LOW_N_CONFIDENCE}

    d["went_over"] = d[EVAL_MARKETS[market]["actual_col"]] > d["line"]

    for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
        sel = d[(d["prob_over"] >= lo) & (d["prob_over"] < hi)] if hi < 1.0 else \
              d[(d["prob_over"] >= lo) & (d["prob_over"] <= hi)]
        n = len(sel)
        if n == 0:
            rows.append({"band": f"{lo:.1f}–{hi:.1f}", "n": 0, "predicted": None,
                         "realized": None, "gap": None, "se2": None, "se_valid": False,
                         "thin": True})
            continue
        pred = float(sel["prob_over"].mean())
        real = float(sel["went_over"].mean())
        valid = min(n * real, n * (1 - real)) >= 5
        se = math.sqrt(real * (1 - real) / n) if valid else None
        rows.append({
            "band": f"{lo:.1f}–{hi:.1f}", "n": n,
            "predicted": pred, "realized": real, "gap": real - pred,
            "se2": (2 * se) if se is not None else None,
            "se_valid": valid,
            "thin": n < MIN_CELL_N,
        })
    return {"rows": rows, "n": len(d), "low_n": len(d) < LOW_N_CONFIDENCE,
            "low_n_threshold": LOW_N_CONFIDENCE}


def pooled_confidence(season, frames):
    """Confidence calibration pooled across all three markets."""
    parts = []
    for market, df in frames.items():
        if df.empty or "prob_over" not in df.columns:
            continue
        ac = EVAL_MARKETS[market]["actual_col"]
        d = df[(df["status"] == "GRADED") & df["prob_over"].notna()
               & df["line"].notna() & df[ac].notna()][["prob_over", "line", ac]].copy()
        if d.empty:
            continue
        d = d.rename(columns={ac: "actual"})
        d["status"] = "GRADED"
        parts.append(d)
    if not parts:
        return {"rows": [], "n": 0, "low_n": True,
                "low_n_threshold": LOW_N_CONFIDENCE}

    pooled = pd.concat(parts, ignore_index=True)
    pooled["went_over"] = pooled["actual"] > pooled["line"]

    rows = []
    for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
        sel = pooled[(pooled["prob_over"] >= lo) & (pooled["prob_over"] < hi)] \
            if hi < 1.0 else \
            pooled[(pooled["prob_over"] >= lo) & (pooled["prob_over"] <= hi)]
        n = len(sel)
        if n == 0:
            rows.append({"band": f"{lo:.1f}–{hi:.1f}", "n": 0, "predicted": None,
                         "realized": None, "gap": None, "se2": None, "se_valid": False,
                         "thin": True})
            continue
        pred = float(sel["prob_over"].mean())
        real = float(sel["went_over"].mean())
        valid = min(n * real, n * (1 - real)) >= 5
        se = math.sqrt(real * (1 - real) / n) if valid else None
        rows.append({"band": f"{lo:.1f}–{hi:.1f}", "n": n, "predicted": pred,
                     "realized": real, "gap": real - pred,
                     "se2": (2 * se) if se is not None else None,
                     "se_valid": valid, "thin": n < MIN_CELL_N})
    return {"rows": rows, "n": len(pooled),
            "low_n": len(pooled) < LOW_N_CONFIDENCE,
            "low_n_threshold": LOW_N_CONFIDENCE}
