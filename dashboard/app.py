"""Read-only Flask dashboard for the weekly review queue.

Run:  venv\\Scripts\\python.exe -m dashboard.app [--season 2026] [--week 1] [--port 5000]

This app opens parquet logs and renders them. It never writes to data/, never
triggers a model run, and never edits an override file.
"""

import argparse
import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import data as D  # noqa: E402

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


@app.route("/")
def index():
    season = request.args.get("season", DEFAULT_SEASON, type=int)
    week = request.args.get("week", DEFAULT_WEEK, type=int)
    label = request.args.get("label") or None
    return render_template("index.html", **build_context(season, week, label))


def main():
    global DEFAULT_SEASON, DEFAULT_WEEK

    ap = argparse.ArgumentParser(description="Read-only prediction review dashboard.")
    ap.add_argument("--season", type=int, default=DEFAULT_SEASON)
    ap.add_argument("--week", type=int, default=DEFAULT_WEEK)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dump", action="store_true",
                    help="print the rendered data server-side and exit (no server)")
    args = ap.parse_args()

    DEFAULT_SEASON, DEFAULT_WEEK = args.season, args.week

    if args.dump:
        dump(args.season, args.week)
        return

    print("=" * 72)
    print("  NFL prop review dashboard — READ ONLY")
    print("  This app never writes to data/, never runs a model, never edits an override.")
    print("=" * 72)
    print(f"  default view : {args.season} week {args.week}")
    print(f"  serving on   : http://{args.host}:{args.port}/")
    print(f"  other weeks  : http://{args.host}:{args.port}/?season=2025&week=8")
    print(f"  labeled log  : http://{args.host}:{args.port}/?season=2026&week=1&label=props_test")
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


if __name__ == "__main__":
    main()
