"""Read-only Flask dashboard: weekly review queue + evaluation.

Run:  venv\\Scripts\\python.exe -m dashboard.app [--season 2026] [--week 1] [--port 5000]

Two pages:
  /            Review     — the forward-looking board for an upcoming week.
  /evaluation  Evaluation — backward-looking accuracy, from *_grades.parquet.

This app opens parquet logs and renders them. It never writes to data/, never
triggers a model run, never triggers a GRADE, and never edits an override file.
The evaluation page reads grade files that src.models.evaluate* already wrote;
if a week has not been graded it says so rather than grading it.
"""

import argparse
import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import data as D  # noqa: E402
from dashboard import evaluation as E  # noqa: E402

app = Flask(__name__)

DEFAULT_SEASON = 2026
DEFAULT_WEEK = 1


def build_context(season, week, label=None):
    """Assemble everything the page needs. Never raises on missing data."""
    markets = D.load_all_markets(season, week, label)
    starters, starters_error = D.qb_starters(season, week)
    sequences = D.team_game_sequences()
    prices = D.market_prices()

    board = D.prop_board(markets, prices)
    stale = D.staleness_rows(markets, sequences)
    attention = D.needs_attention(season, week, markets, starters, board)

    sources = []
    for m in markets.values():
        gen = "—"
        df = m["df"]
        if not df.empty and "generated_at_utc" in df.columns:
            try:
                gen = str(df["generated_at_utc"].max())[:19]
            except Exception:                                  # noqa: BLE001
                pass
        sources.append({
            "market": m["market_label"],
            "filename": m["filename"] or "—",
            "kind": m["kind"],
            "note": m["note"],
            "rows": len(df),
            "generated": gen,
            "error": m["error"],
        })

    any_log = any(m["kind"] != "missing" for m in markets.values())

    return {
        "season": season,
        "week": week,
        "label": label,
        "sources": sources,
        "any_log": any_log,
        "attention": attention,
        "board": board,
        "starters": starters.to_dict("records") if not starters.empty else [],
        "starters_error": starters_error,
        "stale": stale,
    }


def build_evaluation_context(season, week=None, tab="last_week"):
    """Assemble the evaluation page. Never raises on missing or ungraded data."""
    weeks = E.graded_weeks(season)
    latest = weeks[-1] if weeks else None

    # Default to the most recent graded week; honour an explicit earlier pick.
    if week is None or week not in weeks:
        week = latest

    ctx = {
        "season": season,
        "week": week,
        "tab": tab,
        "graded_weeks": weeks,
        "latest_graded_week": latest,
        "any_grades": bool(weeks),
        "markets": [],
        "season_rows": [],
        "wow": {},
        "buckets": {},
        "confidence": {},
        "pooled": None,
        "season_totals": None,
    }

    if not weeks:
        return ctx

    if tab == "last_week" and week is not None:
        grades = E.load_all_grades(season, week)
        for m in E.EVAL_MARKETS:
            summary = E.market_summary(season, week, m, grades[m])
            best, worst = E.best_worst(season, week, m, grades[m], n=10)
            summary["best"] = best
            summary["worst"] = worst
            summary["availability"] = E.availability_rows(
                season, week, m, grades[m], n=10)
            ctx["markets"].append(summary)
        return ctx

    # Season tab: cumulative across every graded week of this season.
    frames = {}
    tot_w = tot_n = tot_ou = tot_ou_w = 0
    for m in E.EVAL_MARKETS:
        df, mweeks = E.load_season_grades(season, m)
        frames[m] = df
        s = E.season_summary(season, m, df, mweeks)
        ctx["season_rows"].append(s)
        ctx["wow"][m] = E.week_over_week(season, m)
        ctx["buckets"][m] = E.cumulative_buckets(season, m, df)
        ctx["confidence"][m] = E.confidence_buckets(season, m, df)
        tot_w = max(tot_w, s["n_weeks"])
        tot_n += s["n_graded"]
        tot_ou += s["ou_n"]
        tot_ou_w += s["ou_wins"]

    ctx["pooled"] = E.pooled_confidence(season, frames)
    ctx["season_totals"] = {
        "n_weeks": tot_w, "n_graded": tot_n, "ou_n": tot_ou, "ou_wins": tot_ou_w,
        "ou_rate": (tot_ou_w / tot_ou) if tot_ou else None,
    }
    return ctx


@app.route("/")
def index():
    season = request.args.get("season", DEFAULT_SEASON, type=int)
    week = request.args.get("week", DEFAULT_WEEK, type=int)
    label = request.args.get("label") or None
    ctx = build_context(season, week, label)
    ctx["nav_active"] = "review"
    return render_template("index.html", **ctx)


@app.route("/evaluation")
def evaluation():
    season = request.args.get("season", DEFAULT_SEASON, type=int)
    week = request.args.get("week", type=int)
    tab = request.args.get("tab", "last_week")
    if tab not in ("last_week", "season"):
        tab = "last_week"
    ctx = build_evaluation_context(season, week, tab)
    ctx["nav_active"] = "evaluation"
    return render_template("evaluation.html", **ctx)


def main():
    global DEFAULT_SEASON, DEFAULT_WEEK

    ap = argparse.ArgumentParser(description="Read-only prediction review dashboard.")
    ap.add_argument("--season", type=int, default=DEFAULT_SEASON)
    ap.add_argument("--week", type=int, default=DEFAULT_WEEK)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dump", action="store_true",
                    help="print the rendered review data server-side and exit")
    ap.add_argument("--dump-evaluation", action="store_true",
                    help="print the rendered evaluation data (both tabs) and exit")
    args = ap.parse_args()

    DEFAULT_SEASON, DEFAULT_WEEK = args.season, args.week

    if args.dump:
        dump(args.season, args.week)
        return

    if args.dump_evaluation:
        # --week defaults to 1 for the review page, but the evaluation page
        # defaults to the LATEST graded week. Only honour --week here if it was
        # actually typed, so the dump matches what /evaluation would render.
        typed = any(a == "--week" or a.startswith("--week=") for a in sys.argv)
        dump_evaluation(args.season, args.week if typed else None)
        return

    print("=" * 72)
    print("  NFL prop review dashboard — READ ONLY")
    print("  This app never writes to data/, never runs a model, never edits an override.")
    print("=" * 72)
    print(f"  default view : {args.season} week {args.week}")
    print(f"  serving on   : http://{args.host}:{args.port}/")
    print(f"  other weeks  : http://{args.host}:{args.port}/?season=2025&week=8")
    print(f"  labeled log  : http://{args.host}:{args.port}/?season=2026&week=1&label=props_test")
    print(f"  evaluation   : http://{args.host}:{args.port}/evaluation?season={args.season}")
    print("  Ctrl-C to stop.")
    print("=" * 72)
    app.run(host=args.host, port=args.port, debug=False)


def dump(season, week, label=None):
    """Server-side sanity dump of exactly what the page would render."""
    ctx = build_context(season, week, label)
    print("=" * 78)
    print(f"RENDERED DATA — {season} week {week}" + (f" (label={label})" if label else ""))
    print("=" * 78)

    print("\n[SOURCES]")
    for s in ctx["sources"]:
        err = f"  ERROR: {s['error']}" if s["error"] else ""
        print(f"  {s['market']:12s} {s['kind']:9s} rows={s['rows']:4d}  {s['filename']}")
        print(f"               generated {s['generated']}  ({s['note']}){err}")
    if not ctx["any_log"]:
        print("\n  EMPTY STATE: no prediction log found for this week.")

    print(f"\n[1] NEEDS ATTENTION — {len(ctx['attention'])} items")
    for a in ctx["attention"]:
        print(f"  [{a['severity']:8s}] {a['category']:14s} {a['subject']}")
        print(f"               why: {a['why']}")
    if not ctx["attention"]:
        print("  (nothing flagged)")

    print(f"\n[2] PROP BOARD — {len(ctx['board'])} priced rows")
    if ctx["board"]:
        print(f"  {'player':22s}{'team':5s}{'market':12s}{'pred':>8s}{'line':>8s}"
              f"{'ours':>7s}{'mkt':>7s}{'diff':>8s}{'books':>6s}")
        for r in ctx["board"]:
            print(f"  {str(r['player'])[:21]:22s}{str(r['team']):5s}{r['market']:12s}"
                  f"{r['prediction']:8.1f}{r['line']:8.1f}{r['prob_over']:7.3f}"
                  f"{(r['mkt_prob'] if r['mkt_prob'] is not None else float('nan')):7.3f}"
                  f"{(r['diff'] if r['diff'] is not None else float('nan')):8.3f}"
                  f"{str(r['n_books'] if r['n_books'] is not None else '-'):>6s}")
    else:
        print("  (no priced rows — no prop lines attached to this week's logs)")

    print(f"\n[3] QB STARTERS — {len(ctx['starters'])} rows")
    if ctx["starters_error"]:
        print(f"  ERROR: {ctx['starters_error']}")
    for r in ctx["starters"]:
        flag = "  <-- SKIP" if r["source"] == "skip" else (
            f"  <-- latest_team={r['latest_team']}" if r.get("team_mismatch") else "")
        print(f"  {str(r['team']):5s}{str(r['qb_name'] or '—'):20s}{str(r['source']):9s}"
              f"{str(r['as_of']):12s}{str(r.get('window_starts')):10s}"
              f"{str(r.get('latest_team') or '—'):5s}{flag}")

    print(f"\n[4] ROSTER STALENESS — {len(ctx['stale'])} rows (oldest first)")
    for r in ctx["stale"][:25]:
        gb = r["games_back"]
        print(f"  {str(r['as_of']):12s}{str(r['player'])[:21]:22s}{str(r['team']):5s}"
              f"{r['market']:12s} games_back={gb if gb is not None else '?':>3}  "
              f"(basis: {r['slice_basis']})")
    if len(ctx["stale"]) > 25:
        print(f"  ... {len(ctx['stale']) - 25} more")
    if not ctx["stale"]:
        print("  (no rows with as_of — logs may predate the as_of column)")


def _pct(v, nd=1):
    return "—" if v is None else f"{v * 100:.{nd}f}%"


def _num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def dump_evaluation(season, week=None):
    """Server-side dump of exactly what the evaluation page would render."""
    from dashboard import evaluation as EV

    # ---------------------------------------------------------- TAB 1
    ctx = build_evaluation_context(season, week, "last_week")
    print("=" * 100)
    print(f"EVALUATION — season {season}")
    print("=" * 100)
    print("READ-ONLY: reads *_grades.parquet only. Never writes, never runs, never grades.")
    print(f"graded weeks available: {ctx['graded_weeks'] or '(none)'}")

    if not ctx["any_grades"]:
        print("\n  EMPTY STATE: no graded weeks for this season yet.")
        return

    print(f"\n{'=' * 100}")
    print(f"TAB 1 — LAST WEEK  (showing week {ctx['week']}, latest graded = "
          f"{ctx['latest_graded_week']})")
    print("=" * 100)

    for m in ctx["markets"]:
        print(f"\n{'-' * 100}")
        print(f"{m['label'].upper()}")
        print(f"{'-' * 100}")
        if not m["exists"]:
            print(f"  NOT GRADED — no file {m['filename']}")
            continue
        if m["error"]:
            print(f"  ERROR: {m['error']}")
            continue

        print(f"  source : {m['filename']}")
        print(f"           generated {m['generated']}   file mtime {m['mtime']}")

        print(f"\n  [CARDS]")
        print(f"    rows graded             : {m['n_graded']}  (of {m['n_rows']} on the board)")
        print(f"    {m['point_label']:24s}: {_num(m['mae'])}   n={m['n_slice']}   "
              f"[prop slice: {m['slice_note']}]")
        bias = m["bias"]
        direction = "" if bias is None else (
            "over-predicted" if bias > 0 else "under-predicted")
        sign = "" if bias is None else ("+" if bias > 0 else "")
        print(f"    bias                    : {sign}{_num(bias)}   "
              f"({'+ = over-predicted' if bias is not None else 'n/a'}"
              f"{'; we ' + direction if direction else ''})")
        print(f"    over/under              : {m['ou_wins']}-{m['ou_n'] - m['ou_wins']}"
              f"  ({_pct(m['ou_rate'])})   n={m['ou_n']}")
        print(f"    {m['accuracy_label']:24s}: "
              f"{m['accuracy_num']}/{m['accuracy_den']} = {_num(m['accuracy'], 3)}")
        print(f"    all-GRADED MAE / bias   : {_num(m['mae_all'])} / {_num(m['bias_all'])}"
              f"   n={m['n_graded']}")

        print(f"\n  [CLASS COUNTS]  {m['classes']}")
        print(f"    availability-event rows on the board: {m['n_availability']}")

        slice_hdr = (f"prop slice: {m['slice_note']}, n={m['n_slice']}"
                     if m["has_slice"] else f"{m['slice_note']}, n={m['n_slice']}")
        for kind, rowset in (("BEST", m["best"]), ("WORST", m["worst"])):
            direction = "smallest" if kind == "BEST" else "largest"
            print(f"\n  [{kind} {len(rowset)} ({direction} |error|, {slice_hdr})]")
            if not rowset:
                print("    (none)")
                continue
            print(f"    {'':1s} {'player':22s}{'tm':4s}{'pred':>8s}{'actual':>8s}"
                  f"{'error':>9s}{'line':>8s}{'p(over)':>9s}  {'bucket':14s}")
            for r in rowset:
                mark = "!" if r["availability"] else " "
                print(f"    {mark} {str(r['player'])[:21]:22s}{str(r['team'] or '—'):4s}"
                      f"{_num(r['pred'], 1):>8s}{_num(r['actual'], 1):>8s}"
                      f"{_num(r['error'], 1):>9s}{_num(r['line'], 1):>8s}"
                      f"{_num(r['prob_over'], 3):>9s}  {str(r['bucket'] or '—'):14s}")
                if r["availability"]:
                    print(f"      ^ AVAILABILITY EVENT — {r['availability_reason']}")
        av = m.get("availability") or []
        print(f"\n  [AVAILABILITY EVENTS — {m['n_availability_excluded']} rows never "
              f"graded; showing top {len(av)} by prediction. EXCLUDED from MAE.]")
        if not av:
            print("    (none — every board row graded)")
        else:
            print(f"    {'player':22s}{'tm':4s}{'pred':>8s}{'line':>8s}"
                  f"{'p(over)':>9s}  {'status':14s}")
            for r in av:
                print(f"    {str(r['player'])[:21]:22s}{str(r['team'] or '—'):4s}"
                      f"{_num(r['pred'], 1):>8s}{_num(r['line'], 1):>8s}"
                      f"{_num(r['prob_over'], 3):>9s}  {str(r['status']):14s}")
            print("    ^ no actual and no error, so these cannot be ranked by miss "
                  "size.\n      Roster-resolution failures, not model failures.")
        print(f"    plus {m['n_availability_flagged']} GRADED row(s) flagged '!' "
              f"inside the tables above — {m['n_availability']} in total.")

        print(f"\n  LEGEND: '!' = availability event, not model error. Driven by the "
              f"grader's\n          status column (DID_NOT_PLAY / NO_STATS / "
              f"WRONG_STARTER); for QB also a\n          starter who did not finish as "
              f"his team's passer (derived, see evaluation.py).")

    # ---------------------------------------------------------- TAB 2
    ctx = build_evaluation_context(season, None, "season")
    print(f"\n\n{'=' * 100}")
    print(f"TAB 2 — SEASON  (cumulative across every graded week of {season})")
    print("=" * 100)

    tot = ctx["season_totals"]
    print(f"\n  weeks graded: {tot['n_weeks']}   graded rows across markets: "
          f"{tot['n_graded']}   pooled over/under: {tot['ou_wins']}-"
          f"{tot['ou_n'] - tot['ou_wins']} ({_pct(tot['ou_rate'])}) n={tot['ou_n']}")
    if tot["n_weeks"] <= 2:
        print(f"  ** n WARNING: {tot['n_weeks']} graded week(s). Rows within a week share "
              f"one league-wide\n     game environment, so effective n is far below these "
              f"counts. Logging only.")

    print(f"\n  [PER-MARKET SUMMARY]")
    print(f"    {'market':14s}{'weeks':>6s}{'rows':>7s}{'graded':>8s}{'slice n':>9s}"
          f"{'MAE':>8s}{'bias':>8s}{'O/U':>12s}{'rate':>8s}{'avail':>8s}")
    for s in ctx["season_rows"]:
        print(f"    {s['label']:14s}{s['n_weeks']:>6d}{s['n_rows']:>7d}{s['n_graded']:>8d}"
              f"{s['n_slice']:>9d}{_num(s['mae']):>8s}{_num(s['bias']):>8s}"
              f"{str(str(s['ou_wins']) + '-' + str(s['ou_n'] - s['ou_wins'])):>12s}"
              f"{_pct(s['ou_rate']):>8s}{_num(s['accuracy'], 3):>8s}")
    print(f"    (MAE and bias are PROP-SLICE; bias + = over-predicted)")

    for s in ctx["season_rows"]:
        m = s["market"]
        print(f"\n{'-' * 100}")
        print(f"{s['label'].upper()} — week over week   [slice: {s['slice_note']}]")
        print(f"{'-' * 100}")
        rows = ctx["wow"][m]
        if not rows:
            print("  (no graded weeks)")
        else:
            print(f"  {'wk':>4s}{'rows':>7s}{'graded':>8s}{'slice n':>9s}{'MAE':>8s}"
                  f"{'bias':>9s}{'O/U':>10s}{'rate':>8s}{'avail':>18s}")
            for r in rows:
                ou = f"{r['ou_wins']}-{r['ou_n'] - r['ou_wins']}"
                av = (f"{r['accuracy_num']}/{r['accuracy_den']}="
                      f"{_num(r['accuracy'], 3)}") if r["accuracy"] is not None else "—"
                print(f"  {r['week']:>4d}{r['n_rows']:>7d}{r['n_graded']:>8d}"
                      f"{r['n_slice']:>9d}{_num(r['mae']):>8s}{_num(r['bias']):>9s}"
                      f"{ou:>10s}{_pct(r['ou_rate']):>8s}{av:>18s}")

        b = ctx["buckets"][m]
        print(f"\n  [CUMULATIVE QUANTILE BUCKETS — prop slice only]  n={b['n']}"
              f"   boundary ties split: {b['n_ties']}")
        if b["low_n"]:
            print(f"  ** n WARNING: n={b['n']} is below {b['low_n_threshold']}. This table "
                  f"is mostly sampling noise.")
        if b["n"]:
            print(f"    {'bucket':12s}{'weight':>8s}{'observed':>10s}{'expected':>10s}"
                  f"{'deviation':>11s}{'+/-2SE':>9s}   {'verdict':s}")
            for r in b["rows"]:
                verdict = ("outside noise" if r["outside_noise"]
                           else ("too few to test" if not r.get("se_valid")
                                 else ("thin cell" if r["thin"]
                                       else "within noise")))
                print(f"    {r['bucket']:12s}{r['weight']:>8.1f}{_pct(r['observed']):>10s}"
                      f"{_pct(r['expected']):>10s}"
                      f"{(('+' if (r['deviation'] or 0) > 0 else '') + _pct(r['deviation'])):>11s}"
                      f"{(_pct(r['se2']) if r.get('se_valid') else 'n/a'):>9s}"
                      f"   {verdict}")
        else:
            print("    (no rows in the prop slice)")

        c = ctx["confidence"][m]
        print(f"\n  [OVER/UNDER BY p(over) CONFIDENCE BAND]  n={c['n']}")
        if c["low_n"]:
            print(f"  ** n WARNING: n={c['n']} is below {c['low_n_threshold']}. Not a "
                  f"threshold input yet.")
        if c["rows"]:
            print(f"    {'band':>10s}{'n':>6s}{'predicted':>11s}{'realized':>10s}"
                  f"{'gap':>9s}{'+/-2SE':>9s}   note")
            for r in c["rows"]:
                note = "empty" if r["n"] == 0 else ("thin" if r["thin"] else "")
                print(f"    {r['band']:>10s}{r['n']:>6d}{_pct(r['predicted']):>11s}"
                      f"{_pct(r['realized']):>10s}"
                      f"{(('+' if (r['gap'] or 0) > 0 else '') + _pct(r['gap'])):>9s}"
                      f"{(_pct(r['se2']) if r.get('se_valid') else 'n/a'):>9s}"
                      f"   {note}")

    pooled = ctx["pooled"]
    print(f"\n{'-' * 100}")
    print(f"POOLED ACROSS ALL THREE MARKETS — over/under by p(over) band")
    print(f"{'-' * 100}")
    print(f"  n={pooled['n']}")
    if pooled["low_n"]:
        print(f"  ** n WARNING: n={pooled['n']} is below {pooled['low_n_threshold']}. "
              f"This is the direct input to the\n     week 5-6 threshold decision and it "
              f"is not yet usable for that purpose.")
    if pooled["rows"]:
        print(f"    {'band':>10s}{'n':>6s}{'predicted':>11s}{'realized':>10s}"
              f"{'gap':>9s}{'+/-2SE':>9s}   note")
        for r in pooled["rows"]:
            note = "empty" if r["n"] == 0 else ("thin" if r["thin"] else "")
            print(f"    {r['band']:>10s}{r['n']:>6d}{_pct(r['predicted']):>11s}"
                  f"{_pct(r['realized']):>10s}"
                  f"{(('+' if (r['gap'] or 0) > 0 else '') + _pct(r['gap'])):>9s}"
                  f"{(_pct(r['se2']) if r.get('se_valid') else 'n/a'):>9s}"
                  f"   {note}")

    print(f"\n  NOTE: no edge column, no market comparison, and no highlighting of "
          f"profitable-looking\n  rows anywhere on this page. Pure logging until the "
          f"threshold is set.")


if __name__ == "__main__":
    main()
