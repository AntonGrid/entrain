"""``python -m entrain_infer`` -- run the M1 bench and write the artefacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import bench, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="entrain-infer", description=__doc__)
    parser.add_argument("--criteria", default=None, help="path to criteria.json")
    parser.add_argument("--out", default=None, help="path to results.json")
    parser.add_argument("--no-report", action="store_true", help="skip the report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    criteria = bench.load_criteria(args.criteria)
    results = bench.run_all(criteria, verbose=not args.quiet)
    path = bench.save_results(results, Path(args.out) if args.out else None)
    print(f"[bench] results -> {path}")
    if not args.no_report:
        written = report.write_all(results)
        for item in written:
            print(f"[report] {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
