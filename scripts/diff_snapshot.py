"""Diff the current feature matrices and raw layer against a dated snapshot.

Step 4 of the Tuesday cadence (RUNBOOK §2). The snapshot is taken BEFORE
re-ingesting; this compares what is on disk now against it, so the weekly
refresh can be confirmed purely additive rather than assumed to be.

What it checks, per matrix:
  * rows ADDED / REMOVED, by key rather than by row count -- a count alone
    cannot distinguish "10 added" from "15 added and 5 dropped".
  * which season/week the added rows belong to, so growth can be confirmed
    to be the week just played and nothing else.
  * value drift on the SHARED spine. This is the one that matters: an
    nflverse refresh can silently rewrite historical values, and that is
    invisible to any row count. See the §2 note on the August 2026
    quantile-artifact drift that could not be diagnosed for want of a
    pre-refresh copy.

Raw files are compared on row count and column set only -- players.parquet
has no stable natural key across refreshes, and the thing worth catching
there (status flips driving players off the board) shows up as a row-count
or value change that the board diff surfaces anyway.

Usage:
    venv\Scripts\python.exe scripts\diff_snapshot.py --date YYYY-MM-DD
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = PROJECT_ROOT / "data" / "features"
RAW = PROJECT_ROOT / "data" / "raw"
BACKUPS = PROJECT_ROOT / "data" / "_feature_backups"

# Natural keys. A matrix row is identified by these, not by position.
MATRICES = {
    "qb_matrix": ["season", "week", "team"],
    "receiving_matrix": ["season", "week", "team", "gsis_id"],
    "rushing_matrix": ["season", "week", "team", "gsis_id"],
}

# Tolerance for "unchanged" on floats. Tight enough that a real recomputation
# shows up, loose enough that parquet round-tripping does not.
RTOL = ATOL = 1e-9


def diff_matrix(name, keys, snap_dir):
    new_path = FEATURES / f"{name}.parquet"
    old_path = snap_dir / "features" / f"{name}.parquet"
    if not old_path.exists():
        print(f"  {name}: NOT IN SNAPSHOT — skipped")
        return None

    new = pd.read_parquet(new_path)
    old = pd.read_parquet(old_path)
    keys = [k for k in keys if k in new.columns and k in old.columns]

    print("=" * 94)
    print(f"{name}   {old.shape} -> {new.shape}   ({len(new) - len(old):+d} rows)")
    print("=" * 94)

    old_keys = set(map(tuple, old[keys].values))
    new_keys = set(map(tuple, new[keys].values))
    added, removed = new_keys - old_keys, old_keys - new_keys
    print(f"  key: {keys}")
    print(f"  rows ADDED   : {len(added)}")
    print(f"  rows REMOVED : {len(removed)}")

    if added:
        a = pd.DataFrame(sorted(added), columns=keys)
        print("  added by season/week:")
        for (s, w), n in a.groupby(["season", "week"]).size().items():
            print(f"    {int(s)} wk{int(w):02d}: {n}")
    if removed:
        r = pd.DataFrame(sorted(removed), columns=keys)
        print("  !! REMOVED rows by season:")
        for s, n in r.groupby("season").size().items():
            print(f"    {int(s)}: {n}")
        print(r.head(15).to_string(index=False))

    shared = sorted(old_keys & new_keys)
    o = old.set_index(keys).sort_index().loc[shared]
    n = new.set_index(keys).sort_index().loc[shared]

    drift = {}
    for c in [c for c in o.columns if c in n.columns]:
        x, y = o[c], n[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            d = ~np.isclose(x.astype(float), y.astype(float),
                            rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            d = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
        k = int(np.asarray(d).sum())
        if k:
            drift[c] = k

    print(f"  shared spine rows: {len(shared):,}")
    if drift:
        print(f"  !! HISTORICAL VALUE DRIFT on {len(drift)} column(s): {drift}")
        worst = max(drift, key=drift.get)
        x, y = o[worst], n[worst]
        m = (~np.isclose(x.astype(float), y.astype(float), rtol=RTOL, atol=ATOL,
                         equal_nan=True)
             if pd.api.types.is_numeric_dtype(x)
             else np.asarray(x.astype(str) != y.astype(str)))
        print(f"     worst column '{worst}': seasons affected "
              f"{sorted({i[0] for i in o.index[m]})}")
    else:
        print("  historical value drift: NONE — all shared rows identical")
    print()
    return {"added": len(added), "removed": len(removed), "drift": drift}


def diff_raw(snap_dir):
    print("=" * 94)
    print("RAW LAYER (row counts and column sets)")
    print("=" * 94)
    for p in sorted(RAW.glob("*.parquet")):
        old_path = snap_dir / "raw" / p.name
        if not old_path.exists():
            print(f"  {p.name:24s} NOT IN SNAPSHOT")
            continue
        new, old = pd.read_parquet(p), pd.read_parquet(old_path)
        gained = set(new.columns) - set(old.columns)
        lost = set(old.columns) - set(new.columns)
        note = ""
        if gained:
            note += f"  +cols {sorted(gained)}"
        if lost:
            note += f"  !! -cols {sorted(lost)}"
        print(f"  {p.name:24s} {len(old):>7,} -> {len(new):>7,} "
              f"({len(new) - len(old):+,}){note}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True,
                    help="snapshot folder name under data/_feature_backups (YYYY-MM-DD)")
    args = ap.parse_args()

    snap_dir = BACKUPS / args.date
    if not snap_dir.exists():
        print(f"No snapshot at {snap_dir}")
        print(f"Available: {sorted(p.name for p in BACKUPS.iterdir() if p.is_dir())}")
        sys.exit(1)

    print(f"\nDiffing current state against snapshot {args.date}\n")
    diff_raw(snap_dir)

    results = {}
    for name, keys in MATRICES.items():
        results[name] = diff_matrix(name, keys, snap_dir)

    print("=" * 94)
    print("VERDICT")
    print("=" * 94)
    clean = True
    for name, r in results.items():
        if r is None:
            continue
        flags = []
        if r["removed"]:
            flags.append(f"{r['removed']} ROWS REMOVED")
        if r["drift"]:
            flags.append(f"DRIFT on {len(r['drift'])} column(s)")
        if flags:
            clean = False
            print(f"  {name:20s} +{r['added']:<6d} !! {'; '.join(flags)}")
        else:
            print(f"  {name:20s} +{r['added']:<6d} purely additive, no drift")
    print()
    print("  Purely additive across all three matrices." if clean else
          "  NOT purely additive — read the detail above before rebuilding artifacts.")


if __name__ == "__main__":
    main()
