#!/usr/bin/env python3
"""
One command to refresh everything after a new ft sync.

Stages, in order:
  sync       — `ft sync` (pulls new bookmarks from X via the field-theory CLI)
  articles   — fetch external article bodies (trafilatura)
  x-articles — fetch X-native long-form articles (GraphQL via cookies)
  videos     — trace video bookmarks back to YouTube originals
  graph      — embed all article bodies, write Obsidian-style related links

Usage:
  python sync_all.py                              # all stages
  python sync_all.py --skip sync --skip videos    # opt out of stages
  python sync_all.py --only graph                 # rebuild just one stage
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # .venv/bin/python when invoked from venv

STAGES = {
    "sync": ["ft", "sync"],
    "articles": [PYTHON, str(ROOT / "enrichment" / "enrich_articles.py")],
    "x-articles": [PYTHON, str(ROOT / "enrichment" / "enrich_x_articles.py")],
    "videos": [PYTHON, str(ROOT / "enrichment" / "enrich_videos.py")],
    "graph": [PYTHON, str(ROOT / "enrichment" / "build_graph.py")],
}
ORDER = ["sync", "articles", "x-articles", "videos", "graph"]


def banner(stage: str, cmd: list[str]) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  [{stage}]  {' '.join(shlex.quote(c) for c in cmd)}\n{bar}")


def run(stage: str) -> int:
    cmd = STAGES[stage]
    banner(stage, cmd)
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError as e:
        print(f"  [{stage}] command not found: {e}", file=sys.stderr)
        return 127
    dt = time.time() - t0
    print(f"  [{stage}] exit {rc}  ({dt:.1f}s)")
    return rc


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--skip", action="append", default=[], choices=ORDER,
                    help="Skip a stage (repeatable)")
    ap.add_argument("--only", choices=ORDER,
                    help="Run only this stage")
    args = ap.parse_args()

    if args.only:
        stages_to_run = [args.only]
    else:
        stages_to_run = [s for s in ORDER if s not in args.skip]

    print(f"plan: {' → '.join(stages_to_run)}")
    for stage in stages_to_run:
        rc = run(stage)
        if rc != 0:
            print(f"\nSTOPPED at [{stage}] (exit {rc}). Earlier stages already wrote their output.",
                  file=sys.stderr)
            sys.exit(rc)
    print("\nall stages OK.")


if __name__ == "__main__":
    main()
