"""Player-name -> gsis_id resolver for sportsbook prop lines.

Pure matching logic, no live API calls: it resolves book-supplied player
names against the *scoped* set of players we would actually be predicting
for a given (season, week) -- the receiving + QB prediction rosters -- never
the full players crosswalk. Resolution is team-scoped and exact-normalized
first, then a bounded fuzzy fallback; anything ambiguous or low-confidence
returns "unresolved" (and is logged) rather than guessing a gsis_id.

Ready to drop into the August prop pull: feed it (book_name, team, line)
rows via resolve_prop_lines.
"""

import re
from difflib import SequenceMatcher

import pandas as pd

from src.features.receiving import build_receiving_prediction_features

# Suffix tokens stripped during normalization (after punctuation removal).
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Fuzzy fallback gates: a match must clear the threshold AND beat the
# second-best team candidate by the margin, or it is treated as ambiguous.
FUZZY_THRESHOLD = 0.82
FUZZY_MARGIN = 0.08


def normalize_name(name):
    """Normalize a player name for matching.

    Lowercase; strip apostrophes/periods/hyphens by joining (so "D.K."->"dk",
    "De'Von"->"devon", "Amon-Ra"->"amonra"); replace any other punctuation
    with a space; drop common generational suffixes (Jr/Sr/II/III/IV/V);
    collapse whitespace.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    # Join across apostrophes (ASCII + curly), periods, hyphens.
    s = re.sub(r"[.'’\-]", "", s)
    # Anything else non-alphanumeric becomes a separator.
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def build_candidate_pool(season, week):
    """Scoped set of resolvable players for (season, week).

    Every (gsis_id, name, team, position) we'd actually predict that week:
    the receiving prediction roster (build_receiving_prediction_features) plus
    the resolved QB starters (src.models.predict.resolve_starters). This is the
    ONLY set names are matched against -- not the full crosswalk -- so a name
    can only resolve to someone we are genuinely pricing.
    """
    # Imported here to avoid a module-load dependency on the models package
    # for callers that only need normalize_name.
    from src.models.predict import resolve_starters

    recv = build_receiving_prediction_features(season, week)[
        ["gsis_id", "player", "position", "team"]
    ].rename(columns={"player": "name"})

    _, qb_summary = resolve_starters(season, week)
    qb = qb_summary[["qb_id", "qb_name", "team"]].rename(
        columns={"qb_id": "gsis_id", "qb_name": "name"}
    )
    qb["position"] = "QB"

    pool = pd.concat([recv, qb[["gsis_id", "name", "position", "team"]]], ignore_index=True)
    pool = pool.dropna(subset=["gsis_id", "name", "team"]).drop_duplicates(subset=["gsis_id", "team"])
    pool["norm"] = pool["name"].apply(normalize_name)
    return pool.reset_index(drop=True)


def resolve_player(book_name, team, candidate_pool):
    """Resolve a single (book_name, team) against the candidate pool.

    Returns a dict: {gsis_id, name, method, score, candidates}.
      - method="exact": unique exact normalized match (score 1.0)
      - method="fuzzy": best SequenceMatcher score >= FUZZY_THRESHOLD and
        ahead of the second-best by >= FUZZY_MARGIN (score = best ratio)
      - method="unresolved": no team candidates, an ambiguous tie, or a
        low-confidence best; gsis_id/name None, `candidates` lists the top
        team candidates considered (name, score) for the log.

    Never picks among tied candidates.
    """
    scoped = candidate_pool[candidate_pool["team"] == team]
    nb = normalize_name(book_name)

    if scoped.empty or not nb:
        return {"gsis_id": None, "name": None, "method": "unresolved", "score": None, "candidates": []}

    # (a) exact normalized match
    exact = scoped[scoped["norm"] == nb]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {"gsis_id": row["gsis_id"], "name": row["name"], "method": "exact", "score": 1.0, "candidates": []}
    if len(exact) > 1:
        cands = [(r["name"], 1.0) for _, r in exact.iterrows()]
        return {"gsis_id": None, "name": None, "method": "unresolved", "score": 1.0, "candidates": cands}

    # (b) bounded fuzzy fallback
    scored = sorted(
        ((SequenceMatcher(None, nb, r["norm"]).ratio(), r["name"], r["gsis_id"]) for _, r in scoped.iterrows()),
        key=lambda t: t[0],
        reverse=True,
    )
    best_score, best_name, best_gsis = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    top_candidates = [(name, round(score, 3)) for score, name, _ in scored[:3]]

    if best_score >= FUZZY_THRESHOLD and (best_score - second_score) >= FUZZY_MARGIN:
        return {"gsis_id": best_gsis, "name": best_name, "method": "fuzzy", "score": best_score, "candidates": top_candidates}

    # (c) unresolved
    return {"gsis_id": None, "name": None, "method": "unresolved", "score": best_score, "candidates": top_candidates}


def resolve_prop_lines(lines_df, season, week):
    """Resolve a frame of (book_name, team, line) prop rows for (season, week).

    Returns lines_df augmented with gsis_id / resolved_name / method / score,
    and prints a report: counts by method, every fuzzy match (for eyeball
    verification), and every unresolved name with its top candidates.
    """
    pool = build_candidate_pool(season, week)

    results = [resolve_player(row["book_name"], row["team"], pool) for _, row in lines_df.iterrows()]

    out = lines_df.copy().reset_index(drop=True)
    out["gsis_id"] = [r["gsis_id"] for r in results]
    out["resolved_name"] = [r["name"] for r in results]
    out["method"] = [r["method"] for r in results]
    out["score"] = [r["score"] for r in results]

    counts = out["method"].value_counts()
    print(f"Resolution report ({season} wk{week}, {len(out)} lines):")
    for method in ("exact", "fuzzy", "unresolved"):
        print(f"  {method:<11} {int(counts.get(method, 0))}")

    fuzzy = out[out["method"] == "fuzzy"]
    if len(fuzzy):
        print("\n  Fuzzy matches (verify):")
        for _, r in fuzzy.iterrows():
            print(f"    [{r['team']}] \"{r['book_name']}\" -> {r['resolved_name']} (score {r['score']:.3f})")

    unresolved = [(row, res) for (_, row), res in zip(lines_df.iterrows(), results) if res["method"] == "unresolved"]
    if unresolved:
        print("\n  Unresolved:")
        for row, res in unresolved:
            cand_str = ", ".join(f"{n} ({s})" for n, s in res["candidates"]) or "no team candidates"
            print(f"    [{row['team']}] \"{row['book_name']}\" — top: {cand_str}")

    return out


def resolve_player_event(book_name, home_team, away_team, candidate_pool):
    """Resolve a book name when only the EVENT's two teams are known.

    The per-event prop endpoint carries no per-player team attribution (step
    58), so the candidate scope is both teams. Rather than widening
    resolve_player's scope -- which would silently halve the protection that
    team-scoping provides -- this runs the UNCHANGED resolve_player once per
    team and arbitrates:

      - exactly one team resolves  -> take it, and record resolved_team
      - both teams resolve         -> method "ambiguous_both_teams",
                                      gsis_id None, both candidates logged
      - neither resolves           -> method "unresolved", candidates from
                                      both teams merged for the log

    A name that legitimately matches a player on each side of the same game
    is exactly the case that must NOT be guessed, so it gets its own method
    rather than being folded into "unresolved".
    """
    per_team = {}
    for team in (home_team, away_team):
        if team is None:
            continue
        per_team[team] = resolve_player(book_name, team, candidate_pool)

    hits = {t: r for t, r in per_team.items() if r["method"] in ("exact", "fuzzy")}

    if len(hits) == 1:
        team, res = next(iter(hits.items()))
        return {**res, "resolved_team": team, "team_candidates": per_team}

    if len(hits) > 1:
        cands = [(t, r["name"], round(r["score"], 3) if r["score"] else None)
                 for t, r in hits.items()]
        return {
            "gsis_id": None, "name": None, "method": "ambiguous_both_teams",
            "score": None, "candidates": cands, "resolved_team": None,
            "team_candidates": per_team,
        }

    merged = []
    for t, r in per_team.items():
        merged.extend((f"{t}:{n}", s) for n, s in r["candidates"])
    merged.sort(key=lambda x: x[1], reverse=True)
    best = max((r["score"] for r in per_team.values() if r["score"] is not None), default=None)
    return {
        "gsis_id": None, "name": None, "method": "unresolved",
        "score": best, "candidates": merged[:4], "resolved_team": None,
        "team_candidates": per_team,
    }


def resolve_prop_lines_event(consensus_df, season, week):
    """Resolve event-scoped consensus prop rows for (season, week).

    `consensus_df` is consensus_prop_lines output (carries home_team and
    away_team but no per-player team). Returns it augmented with gsis_id /
    resolved_name / resolved_team / method / score, and prints a report:
    counts by method, every fuzzy match, every ambiguous-both-teams case, and
    every unresolved name with its top candidates from BOTH teams.
    """
    pool = build_candidate_pool(season, week)

    results = [
        resolve_player_event(row["book_name"], row["home_team"], row["away_team"], pool)
        for _, row in consensus_df.iterrows()
    ]

    out = consensus_df.copy().reset_index(drop=True)
    out["gsis_id"] = [r["gsis_id"] for r in results]
    out["resolved_name"] = [r["name"] for r in results]
    out["resolved_team"] = [r["resolved_team"] for r in results]
    out["method"] = [r["method"] for r in results]
    out["score"] = [r["score"] for r in results]

    counts = out["method"].value_counts()
    print(f"Event-scoped resolution report ({season} wk{week}, {len(out)} prop rows):")
    for method in ("exact", "fuzzy", "ambiguous_both_teams", "unresolved"):
        print(f"  {method:<21} {int(counts.get(method, 0))}")

    fuzzy = out[out["method"] == "fuzzy"]
    if len(fuzzy):
        print("\n  Fuzzy matches (verify):")
        for _, r in fuzzy.iterrows():
            print(f"    [{r['resolved_team']}] \"{r['book_name']}\" ({r['market']}) -> "
                  f"{r['resolved_name']} (score {r['score']:.3f})")
    else:
        print("\n  Fuzzy matches: none — every resolved name matched exactly.")

    for row, res in zip(consensus_df.to_dict("records"), results):
        if res["method"] == "ambiguous_both_teams":
            cand_str = ", ".join(f"{t}:{n} ({s})" for t, n, s in res["candidates"])
            print(f"\n  AMBIGUOUS (both teams matched) \"{row['book_name']}\" "
                  f"({row['market']}) — {cand_str}")

    unresolved = [(row, res) for row, res in zip(consensus_df.to_dict("records"), results)
                  if res["method"] == "unresolved"]
    if unresolved:
        print("\n  Unresolved:")
        for row, res in unresolved:
            cand_str = ", ".join(f"{n} ({s})" for n, s in res["candidates"]) or "no candidates"
            print(f"    \"{row['book_name']}\" ({row['market']}) "
                  f"[{row['home_team']}/{row['away_team']}] — top: {cand_str}")

    return out
