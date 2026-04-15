#!/usr/bin/env python3
"""Validate and analyze SFEN training data files.

Checks for malformed lines, reports statistics on score distribution and
game result balance, and optionally detects duplicate positions.

Usage:
    python scripts/validate_sfen.py data/train.sfen
    python scripts/validate_sfen.py data/train.sfen --check-duplicates
    python scripts/validate_sfen.py data/*.sfen --summary
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze SFEN training data files.",
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="One or more SFEN data files to validate.",
    )
    parser.add_argument(
        "--check-duplicates", action="store_true",
        help="Detect duplicate SFEN positions (uses more memory).",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print only the summary, suppress per-file details.",
    )
    return parser.parse_args()


def validate_line(line: str) -> tuple[str | None, float | None, float | None, str | None]:
    """Parse a single training data line.

    Returns (sfen, score, result, error_message).
    On success, error_message is None.
    On failure, sfen/score/result may be None and error_message describes the problem.
    """
    stripped = line.strip()
    if not stripped:
        return None, None, None, "empty line"

    # Strip "sfen " prefix if present.
    content = stripped[5:] if stripped.startswith("sfen ") else stripped

    # The line may be tab-separated (gensfen output) or space-separated.
    # Try tab-split first, fall back to space-split.
    if "\t" in content:
        # Format: "board side hand ply\tscore\tresult"
        tab_parts = content.split("\t")
        if len(tab_parts) < 3:
            return None, None, None, f"expected 3 tab-separated fields, got {len(tab_parts)}"
        sfen_str = tab_parts[0]
        score_str = tab_parts[1]
        result_str = tab_parts[2]
    else:
        # Format: "board side hand ply score result"
        parts = content.split()
        if len(parts) < 6:
            return None, None, None, f"expected >= 6 space-separated fields, got {len(parts)}"
        sfen_str = " ".join(parts[:4])
        score_str = parts[4]
        result_str = parts[5]

    # Validate SFEN structure (4 fields: board, side, hand, ply).
    sfen_parts = sfen_str.split()
    if len(sfen_parts) != 4:
        return None, None, None, f"SFEN should have 4 fields, got {len(sfen_parts)}"

    # Validate side-to-move.
    if sfen_parts[1] not in ("b", "w"):
        return None, None, None, f"invalid side-to-move: {sfen_parts[1]!r}"

    # Validate score.
    try:
        score = float(score_str)
    except ValueError:
        return None, None, None, f"invalid score: {score_str!r}"

    if not math.isfinite(score):
        return None, None, None, f"non-finite score: {score}"

    # Validate result.
    try:
        result = float(result_str)
    except ValueError:
        return None, None, None, f"invalid result: {result_str!r}"

    if result not in (0.0, 0.5, 1.0):
        return None, None, None, f"unexpected result value: {result} (expected 0.0, 0.5, or 1.0)"

    return sfen_str, score, result, None


def validate_file(path: Path, check_duplicates: bool, verbose: bool) -> dict:
    """Validate a single SFEN file and return statistics."""
    stats = {
        "path": str(path),
        "total_lines": 0,
        "valid": 0,
        "malformed": 0,
        "empty": 0,
        "duplicates": 0,
        "errors": [],  # (line_number, error_message) for first N errors
        "scores": [],
        "results": Counter(),
    }

    seen: set[str] | None = set() if check_duplicates else None
    max_errors = 20  # Only report first N errors to avoid flooding output.

    try:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                stats["total_lines"] += 1

                sfen, score, result, error = validate_line(line)

                if error is not None:
                    if error == "empty line":
                        stats["empty"] += 1
                    else:
                        stats["malformed"] += 1
                        if len(stats["errors"]) < max_errors:
                            stats["errors"].append((line_no, error))
                    continue

                # Check for duplicates.
                if seen is not None:
                    if sfen in seen:
                        stats["duplicates"] += 1
                    else:
                        seen.add(sfen)

                stats["valid"] += 1
                stats["scores"].append(score)
                stats["results"][result] += 1

    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        stats["malformed"] = -1
        return stats

    return stats


def print_histogram(scores: list[float], bins: int = 10) -> None:
    """Print a simple text histogram of score distribution."""
    if not scores:
        return

    min_s = min(scores)
    max_s = max(scores)
    if min_s == max_s:
        print(f"  All scores = {min_s:.0f}")
        return

    bin_width = (max_s - min_s) / bins
    counts = [0] * bins
    for s in scores:
        idx = min(int((s - min_s) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    bar_width = 40

    for i in range(bins):
        lo = min_s + i * bin_width
        hi = lo + bin_width
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{lo:>7.0f}, {hi:>7.0f}) {counts[i]:>7}  {bar}")


def print_stats(stats: dict, verbose: bool) -> None:
    """Print validation results for one file."""
    path = stats["path"]
    total = stats["total_lines"]
    valid = stats["valid"]
    malformed = stats["malformed"]
    empty = stats["empty"]
    duplicates = stats["duplicates"]
    scores = stats["scores"]
    results = stats["results"]

    print(f"--- {path} ---")
    print(f"  Total lines:    {total:>10}")
    print(f"  Valid:          {valid:>10}  ({valid / max(total, 1) * 100:.1f}%)")
    if malformed > 0:
        print(f"  Malformed:      {malformed:>10}  ({malformed / max(total, 1) * 100:.1f}%)")
    if empty > 0:
        print(f"  Empty:          {empty:>10}")
    if duplicates > 0:
        print(f"  Duplicates:     {duplicates:>10}  ({duplicates / max(valid, 1) * 100:.1f}% of valid)")

    if verbose and stats["errors"]:
        print("  First errors (up to 20):")
        for line_no, error in stats["errors"]:
            print(f"    line {line_no}: {error}")

    if scores:
        mean = sum(scores) / len(scores)
        sorted_s = sorted(scores)
        median = sorted_s[len(sorted_s) // 2]
        print()
        print("  Score statistics:")
        print(f"    Min:    {min(scores):>10.1f} cp")
        print(f"    Max:    {max(scores):>10.1f} cp")
        print(f"    Mean:   {mean:>10.1f} cp")
        print(f"    Median: {median:>10.1f} cp")
        print()
        print("  Score distribution:")
        print_histogram(scores)

    if results:
        wins = results.get(1.0, 0)
        draws = results.get(0.5, 0)
        losses = results.get(0.0, 0)
        total_r = wins + draws + losses
        print()
        print("  Game results:")
        print(f"    Wins (1.0):   {wins:>10}  ({wins / max(total_r, 1) * 100:.1f}%)")
        print(f"    Draws (0.5):  {draws:>10}  ({draws / max(total_r, 1) * 100:.1f}%)")
        print(f"    Losses (0.0): {losses:>10}  ({losses / max(total_r, 1) * 100:.1f}%)")

    print()


def main() -> None:
    args = parse_args()

    all_stats = []
    has_errors = False

    for path in args.files:
        stats = validate_file(path, args.check_duplicates, verbose=not args.summary)
        all_stats.append(stats)
        if stats["malformed"] > 0:
            has_errors = True

    # Print per-file results.
    if not args.summary:
        for stats in all_stats:
            print_stats(stats, verbose=True)

    # Print aggregate summary if multiple files.
    if len(all_stats) > 1 or args.summary:
        total_lines = sum(s["total_lines"] for s in all_stats)
        total_valid = sum(s["valid"] for s in all_stats)
        total_malformed = sum(s["malformed"] for s in all_stats)
        total_duplicates = sum(s["duplicates"] for s in all_stats)
        all_scores = [s for stats in all_stats for s in stats["scores"]]
        all_results = Counter()
        for stats in all_stats:
            all_results.update(stats["results"])

        print("=== SUMMARY ===")
        print(f"  Files:      {len(all_stats)}")
        print(f"  Total:      {total_lines:>10} lines")
        print(f"  Valid:      {total_valid:>10}  ({total_valid / max(total_lines, 1) * 100:.1f}%)")
        if total_malformed > 0:
            print(f"  Malformed:  {total_malformed:>10}")
        if total_duplicates > 0:
            print(f"  Duplicates: {total_duplicates:>10}")

        if all_scores:
            mean = sum(all_scores) / len(all_scores)
            print(f"  Score mean: {mean:>10.1f} cp")

        if all_results:
            wins = all_results.get(1.0, 0)
            draws = all_results.get(0.5, 0)
            losses = all_results.get(0.0, 0)
            total_r = wins + draws + losses
            print(f"  Results:    {wins} W / {draws} D / {losses} L  "
                  f"({wins / max(total_r, 1) * 100:.0f}%/{draws / max(total_r, 1) * 100:.0f}%/{losses / max(total_r, 1) * 100:.0f}%)")
        print()

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
