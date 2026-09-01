"""Data loading for the read-only dashboard.

READ-ONLY BY CONSTRUCTION: every function here opens files with pandas.read_parquet
or reads a CSV. Nothing in this module writes, and nothing calls a runner.

`resolve_starters` IS imported from the pipeline, but it is itself read-only —
it reads qb_matrix.parquet and starters_override.csv and returns frames.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

MARKETS = {
    "qb": {
        "stem": "qb_pass_yards",
        "label": "QB passing",
        "pred_col": "pred_passing_yards",
        "id_col": "qb_id",
        "name_col": "qb_name",
        "slice_col": None,          # QB has no prop-slice selector
        "slice_min": None,
    },
    "receiving": {
        "stem": "receiving_yards",
        "label": "Receiving",
        "pred_col": "pred_receiving_yards",
        "id_col": "gsis_id",
        "name_col": "player",
        # targets_l8 >= 3 is the prop-relevant slice (evaluate_receiving.
        # PROP_MIN_TARGETS_L8). Logs written before step 74 lack the column;
        # staleness_rows falls back to "has posted line" for those.
        "slice_col": "targets_l8",
        "slice_min": 3,
    },
    "rushing": {
        "stem": "rushing_yards",
        "label": "RB rushing",
        "pred_col": "pred_rushing_yards",
        "id_col": "gsis_id",
        "name_col": "player",
        "slice_col": "carries_l8",
        "slice_min": 8,
    },
}

# From RUNBOOK §9. Static for v1: these are hand-curated judgements, not
# anything the pipeline computes. Scoped by market so a rushing concern does not
# get attached to the same player's receiving row.
WATCHLIST = {
    ("Jawhar Jordan", "rushing"): "4 games of history; inside the rushing prop slice",
    ("Theo Wease", "receiving"): "3 games of history; at the receiving slice boundary",
    ("Malik Willis", "qb"): "6 career games, 1 from 2025 — thinnest QB history on the board",
    ("Malik Nabers", "receiving"): "form from four September games, 13 team-games stale",
    ("Travis Hunter", "receiving"): "canonically CB — gets RB-baseline sigma despite WR usage",
}

# Odds-API market keys -> our market keys, for joining saved prop payloads.
PROP_MARKET_KEYS = {
    "player_pass_yds": "qb",
    "player_reception_yds": "receiving",
    "player_rush_yds": "rushing",
}


# --------------------------------------------------------------------- files

def find_log(season, week, stem, label=None):
    """Locate a prediction log, preferring canonical over labeled.

    Returns (path, kind, note) where kind is "canonical" | "labeled" | "missing".
    If `label` is given, only that exact labeled file is considered.
    Otherwise: canonical first, then the most recently modified labeled file.
    """
    base = f"{season}_w{week:02d}_{stem}"

    if label:
        p = PREDICTIONS_DIR / f"{base}_{label}.parquet"
        if p.exists():
            return p, "labeled", f"explicit label '{label}'"
        return None, "missing", f"no file for label '{label}'"

    canonical = PREDICTIONS_DIR / f"{base}.parquet"
    if canonical.exists():
        return canonical, "canonical", "canonical log"

    candidates = sorted(
        PREDICTIONS_DIR.glob(f"{base}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [c for c in candidates if not c.name.endswith("_grades.parquet")]
    if candidates:
        return candidates[0], "labeled", "no canonical log — showing newest labeled file"
    return None, "missing", "no log found"


def load_market(season, week, market, label=None):
    """Load one market's log. Returns a dict with df + provenance, never raises."""
    cfg = MARKETS[market]
    path, kind, note = find_log(season, week, cfg["stem"], label)
    out = {
        "market": market,
        "market_label": cfg["label"],
        "path": path,
        "filename": path.name if path else None,
        "kind": kind,
        "note": note,
        "df": pd.DataFrame(),
        "error": None,
    }
    if path is None:
        return out
    try:
        out["df"] = pd.read_parquet(path)
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_all_markets(season, week, label=None):
    return {m: load_market(season, week, m, label) for m in MARKETS}


# ------------------------------------------------------------------ helpers

def _american_to_prob(odds):
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    o = float(odds)
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def players_map():
    try:
        pl = pd.read_parquet(RAW_DIR / "players.parquet")
        return pl.drop_duplicates("gsis_id").set_index("gsis_id")
    except Exception:                                          # noqa: BLE001
        return pd.DataFrame()


def market_prices():
    """Book prices for the saved prop payloads, keyed by (norm name, market).

    The prediction logs carry `line` and `prob_over` but NOT the book prices, so
    the de-vigged market probability has to come from the saved raw payloads in
    data/raw/odds/. Read-only: opens saved JSON, makes no API call.

    Returns {} if nothing is on disk or the odds module cannot be imported.
    """
    import json

    odds_dir = RAW_DIR / "odds"
    if not odds_dir.exists():
        return {}
    try:
        from src.ingestion.odds import consensus_prop_lines, parse_event_props
        from src.ingestion.player_resolver import normalize_name
    except Exception:                                          # noqa: BLE001
        return {}

    frames = []
    for p in sorted(odds_dir.glob("props_raw_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
            d = parse_event_props(payload)
            if not d.empty:
                frames.append(d)
        except Exception:                                      # noqa: BLE001
            continue
    if not frames:
        return {}

    allp = pd.concat(frames, ignore_index=True)
    out = {}
    for event_id, g in allp.groupby("event_id"):
        cons = consensus_prop_lines(g)
        for _, r in cons.iterrows():
            mk = PROP_MARKET_KEYS.get(r["market"])
            if not mk:
                continue
            key = (normalize_name(r["book_name"]), mk)
            # keep the freshest row if the same prop appears in several payloads
            prev = out.get(key)
            if prev is None or str(r["last_update"]) > str(prev["last_update"]):
                out[key] = {
                    "over_price": r["over_price"],
                    "under_price": r["under_price"],
                    "n_books": r["n_books"],
                    "line": r["line"],
                    "last_update": r["last_update"],
                }
    return out


def team_game_sequences():
    """team -> ordered list of 'YYYY wkNN' from the receiving spine, for games-back."""
    out = {}
    try:
        rm = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
    except Exception:                                          # noqa: BLE001
        return out
    for team, g in rm.groupby("team"):
        seq = g[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        out[team] = [f"{int(s)} wk{int(w):02d}" for s, w in zip(seq.season, seq.week)]
    return out


def games_back(sequences, team, as_of):
    seq = sequences.get(team, [])
    if not as_of or as_of not in seq:
        return None
    return len(seq) - 1 - seq.index(as_of)


# --------------------------------------------------------------- prop board

def prop_board(markets_data, prices=None):
    """Every priced row across all markets, one table. Display only.

    Book prices come from `prices` (saved payloads) since the logs do not carry
    them; rows with no price match show '—' for market probability rather than
    inventing one.
    """
    try:
        from src.ingestion.player_resolver import normalize_name
    except Exception:                                          # noqa: BLE001
        normalize_name = lambda s: str(s).lower()              # noqa: E731

    prices = prices or {}
    rows = []
    for key, blob in markets_data.items():
        df, cfg = blob["df"], MARKETS[key]
        if df.empty or "line" not in df.columns or "prob_over" not in df.columns:
            continue
        priced = df[df["line"].notna() & df["prob_over"].notna()]
        for _, r in priced.iterrows():
            name = r.get(cfg["name_col"])
            px = prices.get((normalize_name(name), key), {})
            over_p = _american_to_prob(px.get("over_price"))
            under_p = _american_to_prob(px.get("under_price"))
            if not (np.isnan(over_p) or np.isnan(under_p)) and (over_p + under_p) > 0:
                mkt = over_p / (over_p + under_p)
            else:
                mkt = np.nan
            diff = r["prob_over"] - mkt if not np.isnan(mkt) else np.nan
            rows.append({
                "player": name,
                "team": r.get("team"),
                "market": cfg["label"],
                "prediction": round(float(r[cfg["pred_col"]]), 1),
                "line": float(r["line"]),
                "prob_over": round(float(r["prob_over"]), 3),
                "mkt_prob": None if np.isnan(mkt) else round(float(mkt), 3),
                "diff": None if np.isnan(diff) else round(float(diff), 3),
                "n_books": px.get("n_books"),
            })
    return sorted(rows, key=lambda x: (x["diff"] is None, x["diff"]))


# -------------------------------------------------------------- qb starters

def qb_starters(season, week):
    """Read-only: resolve_starters reads qb_matrix + starters_override.csv."""
    try:
        from src.models.predict import resolve_starters
        _, summary = resolve_starters(season, week)
    except Exception as exc:                                   # noqa: BLE001
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"

    pm = players_map()
    if not pm.empty:
        summary["latest_team"] = summary["qb_id"].map(pm["latest_team"])
        summary["status"] = summary["qb_id"].map(pm["status"])
    else:
        summary["latest_team"] = None
        summary["status"] = None
    summary["team_mismatch"] = [
        bool(pd.notna(q) and lt is not None and pd.notna(lt) and lt != t)
        for q, lt, t in zip(summary["qb_id"], summary["latest_team"], summary["team"])
    ]
    # NaN is truthy in Python, so `value or '—'` fallbacks in the template and the
    # dump would render "nan" on SKIP rows. Normalise to None here instead.
    summary = summary.sort_values(["source", "team"]).reset_index(drop=True)
    for col in ("qb_id", "qb_name", "latest_team", "status", "as_of", "window_starts"):
        if col in summary.columns:
            summary[col] = pd.Series(
                [None if pd.isna(v) else v for v in summary[col]],
                dtype=object, index=summary.index,
            )
    return summary, None


# ---------------------------------------------------------------- staleness

def staleness_rows(markets_data, sequences):
    """Rows carrying as_of, oldest first, with games-back where computable."""
    rows = []
    for key, blob in markets_data.items():
        df, cfg = blob["df"], MARKETS[key]
        if df.empty or "as_of" not in df.columns:
            continue
        d = df.copy()
        in_slice = None
        # A log written before its market's slice column existed (receiving
        # gained targets_l8 at step 74) falls back to "has a posted line", and
        # slice_basis must say so rather than naming a column that isn't there.
        basis = "has posted line"
        if cfg["slice_col"] and cfg["slice_col"] in d.columns:
            in_slice = d[cfg["slice_col"]] >= cfg["slice_min"]
            basis = cfg["slice_col"]
        for i, (_, r) in enumerate(d.iterrows()):
            has_line = pd.notna(r.get("line")) if "line" in d.columns else False
            slice_state = (
                bool(in_slice.iloc[i]) if in_slice is not None else None
            )
            # Without a slice column, treat "has a posted line" as the practical filter.
            relevant = slice_state if slice_state is not None else bool(has_line)
            if not relevant:
                continue
            rows.append({
                "player": r.get(cfg["name_col"]),
                "team": r.get("team"),
                "market": cfg["label"],
                "as_of": r.get("as_of"),
                "games_back": games_back(sequences, r.get("team"), r.get("as_of")),
                "slice_basis": basis,
                "prediction": round(float(r[cfg["pred_col"]]), 1),
            })
    return sorted(rows, key=lambda x: (x["as_of"] is None, x["as_of"] or ""))


# ----------------------------------------------------------- needs attention

def needs_attention(season, week, markets_data, starters, board_rows):
    """The review queue. Every row carries a `why`."""
    items = []

    # (a) QB starters: skipped or team-mismatched
    if starters is not None and not starters.empty:
        for _, r in starters.iterrows():
            if r["source"] == "skip":
                items.append({
                    "category": "QB starter",
                    "subject": f"{r['team']} — no starter",
                    "why": "SKIP: starter undecided, team dropped from the board",
                    "severity": "blocking",
                })
            elif r.get("team_mismatch"):
                items.append({
                    "category": "QB starter",
                    "subject": f"{r['team']} — {r['qb_name']}",
                    "why": f"latest_team is {r['latest_team']}, not {r['team']}",
                    "severity": "check",
                })

    # (b) resolved-but-unpriced: a row that has a line but no prob_over
    for key, blob in markets_data.items():
        df, cfg = blob["df"], MARKETS[key]
        if df.empty or "line" not in df.columns:
            continue
        unpriced = df[df["line"].notna() & df.get("prob_over", pd.Series(dtype=float)).isna()]
        for _, r in unpriced.iterrows():
            items.append({
                "category": "Unpriced line",
                "subject": f"{r.get(cfg['name_col'])} ({r.get('team')}) — {cfg['label']}",
                "why": f"line {r['line']} present but no prob_over produced",
                "severity": "check",
            })

    # (c) watchlist players appearing on the board for the market they concern
    seen = set()
    for key, blob in markets_data.items():
        df, cfg = blob["df"], MARKETS[key]
        if df.empty or cfg["name_col"] not in df.columns:
            continue
        for _, r in df.iterrows():
            nm = r.get(cfg["name_col"])
            why = WATCHLIST.get((nm, key))
            if why and (nm, key) not in seen:
                seen.add((nm, key))
                items.append({
                    "category": "Watchlist",
                    "subject": f"{nm} ({r.get('team')}) — {cfg['label']}",
                    "why": why,
                    "severity": "discount",
                })

    order = {"blocking": 0, "check": 1, "discount": 2}
    return sorted(items, key=lambda x: (order.get(x["severity"], 9), x["category"], x["subject"]))
