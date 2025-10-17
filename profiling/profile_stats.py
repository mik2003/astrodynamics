"""profiling.profile_stats

Compact viewer for cProfile pstats files. Designed to be run as:

  py -m cProfile -o profiling\profile.bin -m project
  py -m profiling --units us --lines 50 --timestamp

Textual dumps are written into ./profiling/ by default when using --dump or
--timestamp.
"""

from __future__ import annotations

import argparse
import os
import pstats
from datetime import datetime
from typing import Optional


def main(
    file: str,
    sort: str,
    lines: int,
    dump: Optional[str],
    units: str,
    precision: int,
) -> None:
    try:
        p = pstats.Stats(file)
    except FileNotFoundError:
        raise SystemExit(
            f"Profile file '{file}' not found. Create it with:\n  py -m cProfile -o {os.path.join('profiling','profile.bin')} -m project"
        )

    unit_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if units not in unit_multipliers:
        raise SystemExit(
            f"Unsupported units '{units}', choose from {list(unit_multipliers)}"
        )
    mul = unit_multipliers[units]

    p.strip_dirs()
    stats = getattr(p, "stats", {})
    items = list(stats.items())

    sk = sort.lower()
    if sk == "tottime":
        keyfn = lambda it: it[1][2]
    elif sk == "cumtime":
        keyfn = lambda it: it[1][3]
    elif sk == "ncalls":
        keyfn = lambda it: it[1][0]
    else:
        try:
            p.sort_stats(sk)
            stats = getattr(p, "stats", {})
            items = list(stats.items())
            keyfn = None
        except Exception:
            keyfn = lambda it: it[1][3]

    if keyfn is not None:
        items.sort(key=keyfn, reverse=True)

    header = f"{'ncalls':>10} {'tottime':>{10+precision}} {'percall':>{10+precision}} {'cumtime':>{10+precision}} {'percall':>{10+precision}}  location"
    print(header)
    print("-" * (len(header) + 20))

    fmt = f"{{:.{precision}f}}"
    for func, data in items[:lines]:
        cc, nc, tottime, cumtime, callers = data
        prim = nc if nc else cc
        per_self = tottime / prim if prim else 0.0
        per_cum = cumtime / cc if cc else 0.0

        print(
            f"{str(cc) + ('/' + str(nc) if nc != cc else ''):>10} "
            f"{fmt.format(tottime*mul):>{10+precision}} "
            f"{fmt.format(per_self*mul):>{10+precision}} "
            f"{fmt.format(cumtime*mul):>{10+precision}} "
            f"{fmt.format(per_cum*mul):>{10+precision}}  {func[0]}:{func[1]}({func[2]})"
        )

    if dump:
        os.makedirs(os.path.dirname(dump) or "profiling", exist_ok=True)
        with open(dump, "w", encoding="utf-8") as f:
            ps = pstats.Stats(file, stream=f)
            ps.strip_dirs()
            ps.sort_stats(sort)
            ps.print_stats()
        print(f"Wrote full report to {dump}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="View cProfile stats stored in ./profiling/"
    )
    parser.add_argument(
        "--file",
        default=os.path.join("profiling", "profile.bin"),
        help="path to the cProfile binary (default: profiling/profile.bin)",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        help="sort key for pstats (cumtime, tottime, ncalls, etc.)",
    )
    parser.add_argument(
        "--lines", type=int, default=30, help="number of lines to print"
    )
    parser.add_argument(
        "--dump", default=None, help="path to write full textual report"
    )
    parser.add_argument(
        "--units",
        default="s",
        choices=["s", "ms", "us", "ns"],
        help="units to display times in: s, ms, us, ns",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="decimal places for printed times",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="append a timestamp to the dump filename or generate one if --dump not provided",
    )

    args = parser.parse_args()
    dump = args.dump
    if args.timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dump:
            if not os.path.dirname(dump):
                dump = os.path.join("profiling", dump)
            root, ext = os.path.splitext(dump)
            dump = f"{root}_{ts}{ext}"
        else:
            os.makedirs("profiling", exist_ok=True)
            dump = os.path.join("profiling", f"profile_{ts}.txt")
    else:
        if dump and not os.path.dirname(dump):
            dump = os.path.join("profiling", dump)

    main(
        file=args.file,
        sort=args.sort,
        lines=args.lines,
        dump=dump,
        units=args.units,
        precision=args.precision,
    )


if __name__ == "__main__":
    cli()
