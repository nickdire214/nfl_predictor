"""Hard-assertion tests for src/ingestion/player_resolver.py.

Builds the real 2025 week-8 candidate pool, throws ~25 book-style name
variants at it covering the real hazards (exact, suffix drop, punctuation,
hyphen/space surnames, nickname risk, right-name-wrong-team, and deliberately
unresolvable names), and hard-asserts that every legitimate variant resolves
to the CORRECT gsis_id while unresolvable/low-confidence names return
"unresolved" rather than a wrong match. Also checks the "never pick a tie"
branch with a synthetic ambiguous pool.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_player_resolver.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.ingestion.player_resolver import (
    build_candidate_pool,
    normalize_name,
    resolve_player,
    resolve_prop_lines,
)


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


SEASON, WEEK = 2025, 8
pool = build_candidate_pool(SEASON, WEEK)

# (book_name, team, kind, canonical_name)
#   exact       -> must resolve to canonical via method="exact"
#   safe        -> must be unresolved OR canonical, NEVER a different player
#   unresolved  -> must return method="unresolved"
CASES = [
    ("Ja'Marr Chase", "CIN", "exact", "Ja'Marr Chase"),
    ("Justin Jefferson", "MIN", "exact", "Justin Jefferson"),
    ("Patrick Mahomes", "KC", "exact", "Patrick Mahomes"),
    ("Christian McCaffrey", "SF", "exact", "Christian McCaffrey"),
    ("Travis Kelce", "KC", "exact", "Travis Kelce"),
    ("Xavier Worthy", "KC", "exact", "Xavier Worthy"),
    ("Rashee Rice", "KC", "exact", "Rashee Rice"),
    ("Marquise Brown", "KC", "exact", "Marquise Brown"),
    # suffix drop -> exact after normalization
    ("Michael Pittman", "IND", "exact", "Michael Pittman Jr."),
    ("Michael Pittman Jr", "IND", "exact", "Michael Pittman Jr."),
    ("Michael Pittman Jr.", "IND", "exact", "Michael Pittman Jr."),
    # punctuation -> exact after normalization
    ("DK Metcalf", "PIT", "exact", "D.K. Metcalf"),
    ("D.K. Metcalf", "PIT", "exact", "D.K. Metcalf"),
    ("Jamarr Chase", "CIN", "exact", "Ja'Marr Chase"),
    ("Devon Achane", "MIA", "exact", "De'Von Achane"),
    ("De'Von Achane", "MIA", "exact", "De'Von Achane"),
    ("JuJu Smith-Schuster", "KC", "exact", "JuJu Smith-Schuster"),
    # hyphen-vs-space surname / misspelling -> fuzzy path; correct-or-unresolved
    ("JuJu Smith Schuster", "KC", "safe", "JuJu Smith-Schuster"),
    ("Justin Jeferson", "MIN", "safe", "Justin Jefferson"),
    ("Cristian McCaffrey", "SF", "safe", "Christian McCaffrey"),
    # nickname risk -> must NOT silently become the wrong player
    ("Hollywood Brown", "KC", "safe", "Marquise Brown"),
    # right name, WRONG team -> unresolved (team scoping)
    ("Ja'Marr Chase", "BUF", "unresolved", None),
    ("Patrick Mahomes", "BUF", "unresolved", None),
    # deliberately unresolvable
    ("Johnny Quarterback", "CIN", "unresolved", None),
    ("Zxqwjkl Nonsense", "MIN", "unresolved", None),
    ("Madeup Personnage", "KC", "unresolved", None),
]


def expected_gsis(canonical, team):
    hit = pool[(pool["name"] == canonical) & (pool["team"] == team)]
    assert len(hit) == 1, f"test setup: {canonical!r} ({team}) not unique in pool (found {len(hit)})"
    return hit.iloc[0]["gsis_id"]


# --- (1) Per-case resolution asserts ---
section(f"Per-case resolution ({len(CASES)} book-name variants vs 2025 wk8 pool, size {len(pool)})")

n_exact = n_safe = n_unresolved = 0
for book_name, team, kind, canonical in CASES:
    res = resolve_player(book_name, team, pool)
    want = expected_gsis(canonical, team) if canonical else None

    if kind == "exact":
        assert res["method"] == "exact", f"{book_name!r} ({team}): expected exact, got {res['method']} -> {res['name']}"
        assert res["gsis_id"] == want, f"{book_name!r} ({team}): resolved to {res['name']} ({res['gsis_id']}), want {canonical} ({want})"
        n_exact += 1
        print(f"  EXACT       [{team}] {book_name!r:<26} -> {res['name']}")
    elif kind == "safe":
        # the core safety property: never a DIFFERENT player than the intended one
        assert res["gsis_id"] in (None, want), (
            f"{book_name!r} ({team}): SILENTLY WRONG -> {res['name']} ({res['gsis_id']}), intended {canonical} ({want})"
        )
        n_safe += 1
        verdict = f"{res['method']} -> {res['name']} ({res['score']:.3f})" if res["gsis_id"] else "unresolved (safe)"
        print(f"  SAFE        [{team}] {book_name!r:<26} -> {verdict}")
    else:  # unresolved
        assert res["method"] == "unresolved", f"{book_name!r} ({team}): expected unresolved, got {res['method']} -> {res['name']}"
        assert res["gsis_id"] is None, f"{book_name!r} ({team}): unresolved must have gsis_id None, got {res['gsis_id']}"
        n_unresolved += 1
        print(f"  UNRESOLVED  [{team}] {book_name!r:<26} -> (correctly not matched)")

print(f"\nPASS: {n_exact} exact, {n_safe} safe (no wrong match), {n_unresolved} unresolved — no name silently resolved to the wrong player")


# --- (2) Ambiguous tie -> unresolved (never pick among tied candidates) ---
section("Ambiguous tie: identical normalized names on one team -> unresolved")

amb_pool = pd.DataFrame({
    "gsis_id": ["00-0000001", "00-0000002"],
    "name": ["Mike Williams", "Mike Williams"],
    "team": ["XX", "XX"],
    "position": ["WR", "WR"],
})
amb_pool["norm"] = amb_pool["name"].apply(normalize_name)
amb = resolve_player("Mike Williams", "XX", amb_pool)
assert amb["method"] == "unresolved", f"tie should be unresolved, got {amb['method']} -> {amb['gsis_id']}"
assert amb["gsis_id"] is None, f"tie must not pick a gsis_id, got {amb['gsis_id']}"
print(f"PASS: two 'Mike Williams' on team XX -> {amb['method']} (candidates: {amb['candidates']})")


# --- (3) Full resolution report via resolve_prop_lines ---
section("Full resolution report (resolve_prop_lines)")

lines_df = pd.DataFrame(
    [{"book_name": b, "team": t, "line": 50.5} for (b, t, _, _) in CASES]
)
resolve_prop_lines(lines_df, SEASON, WEEK)


print("\nAll tests passed.")
