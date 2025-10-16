#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import sys
from datetime import datetime
import re

_TS_RE = re.compile(r'\d{8}_\d{6}$')  # YYYYMMDD_HHMMSS


def _decode_ts(s: str) -> datetime:
    return datetime.strptime(s, '%Y%m%d_%H%M%S')


def _closest_ts_ancestor(path: Path) -> Path | None:
    for ancestor in [path, *path.parents]:
        if _TS_RE.fullmatch(ancestor.name):
            return ancestor
    return None


def find_latest_run_dir(model_dir: Path) -> Path | None:
    """
    Find the latest run directory for a model by:
    1) Looking for aggregated_statistics.json files and selecting the nearest
       timestamped ancestor directory with the newest timestamp.
    2) Fallback to newest timestamped directory directly under the model dir.
    3) Final fallback to newest directory by modification time.
    """
    candidates = list(model_dir.rglob('aggregated_statistics.json'))
    items = []
    for c in candidates:
        run_dir = _closest_ts_ancestor(c.parent)
        if run_dir is None:
            continue
        items.append((run_dir, _decode_ts(run_dir.name)))
    if items:
        # choose by decoded timestamp
        return max(items, key=lambda t: t[1])[0]

    # Fallback: search for timestamped directories directly
    ts_dirs = [p for p in model_dir.rglob('*') if p.is_dir() and _TS_RE.fullmatch(p.name)]
    if ts_dirs:
        return max(ts_dirs, key=lambda d: _decode_ts(d.name))

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Copy the latest run per model into saved_named_runs/<run_name>/<model>/<run_dir>."
    )
    parser.add_argument('run_name', help='Name to use under saved_named_runs/')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied without copying')
    parser.add_argument('--source-root', type=Path, default=None,
                        help='Override source root (defaults to tests/results/op_cog next to this script)')
    parser.add_argument('--dest-root', type=Path, default=None,
                        help='Override destination root (defaults to saved_named_runs under source root)')
    parser.add_argument(
        '--use-base-models',
        action='store_true',
        help='Only include models whose names are keys in hal_beam_com.utils.base_models_path'
    )
    args = parser.parse_args()

    selected_names: set[str] | None = None
    if args.use_base_models:
        # Ensure we can import from project src
        here = Path(__file__).resolve()
        for ancestor in [here.parent, *here.parents]:
            src_dir = ancestor / 'src'
            if src_dir.is_dir():
                sys.path.insert(0, str(src_dir))
                break
        try:
            from hal_beam_com.utils import base_models_path  # type: ignore
        except Exception as e:
            parser.error(f"Could not import hal_beam_com.utils.base_models_path: {e}")
        if not isinstance(base_models_path, dict):
            parser.error("hal_beam_com.utils.base_models_path must be a dict-like mapping of model names.")
        selected_names = set(base_models_path.keys())

    op_root = args.source_root or Path(__file__).resolve().parent
    dest_root = args.dest_root or (op_root / 'saved_named_runs')
    target_root = dest_root / args.run_name

    exclude = {'saved_named_runs'}
    model_dirs = [d for d in op_root.iterdir()
                  if d.is_dir() and not d.name.startswith('.') and d.name not in exclude]
    if selected_names is not None:
        model_dirs = [d for d in model_dirs if d.name in selected_names]
        if not model_dirs:
            print(f"[warn] No matching model directories in {op_root} for selected models: {sorted(selected_names)}")

    ops: list[tuple[str, Path, Path]] = []
    ts_name_pairs: list[tuple[datetime, str]] = []

    for mdir in sorted(model_dirs):
        latest = find_latest_run_dir(mdir)
        if latest is None:
            print(f"[skip] No runs found for {mdir.name}")
            continue

        # Track for timespan summary
        ts_name_pairs.append((_decode_ts(latest.name), latest.name))

        dest_dir = target_root / mdir.name / latest.name
        ops.append((mdir.name, latest, dest_dir))

    if not ops:
        print("No runs matched selection; no timespan to report.")
        return

    # Always perform a dry run first
    for model_name, latest, dest_dir in ops:
        print(f"[dry-run] Would copy {latest} -> {dest_dir}")

    # Timespan summary (plan)
    oldest_ts, oldest_name = min(ts_name_pairs, key=lambda x: x[0])
    newest_ts, newest_name = max(ts_name_pairs, key=lambda x: x[0])
    print(f"Timespan of runs to copy: oldest={oldest_name} newest={newest_name}")

    if args.dry_run:
        print("Dry run only. No files copied.")
        return

    # Ask for confirmation before executing
    confirm = input(f"Are you sure you want to copy {len(ops)} runs into {target_root}? Type 'y' to confirm: ").strip().lower()
    if confirm != 'y':
        print("Aborted. No files copied.")
        return

    copied = 0
    for model_name, latest, dest_dir in ops:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(latest, dest_dir, symlinks=True)
        print(f"[ok] Copied {model_name}: {latest.name}")
        copied += 1

    print(f"Timespan of runs copied: oldest={oldest_name} newest={newest_name}")
    print(f"Done. Copied {copied} latest runs into {target_root}")


if __name__ == '__main__':
    main()
