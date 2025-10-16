#!/usr/bin/env python3
"""
Generate an aggregated_statistics.json for a model results directory.

Usage:
  python tests/results/generate_aggregated_statistics.py /path/to/model_results_dir [--force]

Behavior:
- Scans the immediate subdirectories of the provided model results directory.
- Collects each run's run_statistics.json.
- Writes aggregated_statistics.json in the provided directory, containing lists of values per metric key.
- If aggregated_statistics.json already exists, the script will skip writing unless --force is provided.
"""
import argparse
import json
import sys
from pathlib import Path


def find_run_statistics_files(base_dir: Path) -> list[Path]:
    """Return run_statistics.json paths from immediate subdirectories of base_dir."""
    stats_files: list[Path] = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            stats_path = child / "run_statistics.json"
            if stats_path.is_file():
                stats_files.append(stats_path)
    return stats_files


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def aggregate_stats(all_stats: list[dict]) -> dict:
    """
    Create an aggregated dict mapping each key to a list of values across runs.
    Uses the union of keys across all run stats. Missing values become None.
    """
    if not all_stats:
        return {}

    all_keys = set()
    for s in all_stats:
        all_keys.update(s.keys())

    # Sort keys for deterministic output
    aggregated = {k: [s.get(k) for s in all_stats] for k in sorted(all_keys)}
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Generate aggregated_statistics.json for a model results directory.")
    parser.add_argument("results_dir", help="Path to the model results directory (containing per-run subdirectories).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing aggregated_statistics.json if present.")
    args = parser.parse_args()

    base_dir = Path(args.results_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(2)

    aggregated_path = base_dir / "aggregated_statistics.json"
    if aggregated_path.exists() and not args.force:
        print(f"aggregated_statistics.json already exists at: {aggregated_path}\nUse --force to overwrite.")
        sys.exit(0)

    stats_files = find_run_statistics_files(base_dir)
    if not stats_files:
        print(f"No run_statistics.json files found in immediate subdirectories of: {base_dir}", file=sys.stderr)
        sys.exit(1)

    all_stats = []
    for p in stats_files:
        try:
            data = load_json(p)
            if isinstance(data, dict):
                all_stats.append(data)
            else:
                print(f"Warning: Skipping non-dict JSON in {p}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to read {p}: {e}", file=sys.stderr)

    if not all_stats:
        print(f"No valid run_statistics.json contents found under: {base_dir}", file=sys.stderr)
        sys.exit(1)

    aggregated = aggregate_stats(all_stats)

    with aggregated_path.open("w") as f:
        json.dump(aggregated, f, indent=4, sort_keys=True)

    print(f"Aggregated {len(all_stats)} runs into: {aggregated_path}")


if __name__ == "__main__":
    main()
