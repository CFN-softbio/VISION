import json, sys
from pathlib import Path
from typing import Dict, Any
import re
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent    # tests/results
sys.path.append(str(ROOT_DIR))
from tests.results.analyze_statistics import process_nested_dict, format_stats

_TS_RE = re.compile(r'\d{8}_\d{6}$')     # matches “YYYYMMDD_HHMMSS”

def _extract_timestamp(path: Path) -> datetime | None:
    """Return datetime encoded in any directory name of *path*."""
    for part in reversed(path.parts):
        if _TS_RE.fullmatch(part):
            return datetime.strptime(part, '%Y%m%d_%H%M%S')
    return None

def latest_stats_file(model_dir: Path) -> Path | None:
    """
    Choose the aggregated_statistics.json whose ancestor directory name
    contains the most-recent timestamp in the format YYYYMMDD_HHMMSS.
    Falls back to file modification time when no timestamps exist.
    """
    candidates = list(model_dir.rglob('aggregated_statistics.json'))
    if not candidates:
        return None

    with_ts = [(c, _extract_timestamp(c.parent)) for c in candidates]
    ts_candidates = [item for item in with_ts if item[1] is not None]

    if ts_candidates:
        # pick by decoded timestamp
        return max(ts_candidates, key=lambda t: t[1])[0]

    return None

def main():
    op_root = Path(__file__).resolve().parent          # tests/results/op_cog
    model_dirs = [d for d in op_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    summaries: Dict[str, str] = {}
    for mdir in model_dirs:
        stats_file = latest_stats_file(mdir)
        if not stats_file:
            continue
        with stats_file.open() as f:
            raw = json.load(f)
        processed = process_nested_dict(raw)
        summaries[mdir.name] = format_stats(processed)

    if not summaries:
        print('No aggregated_statistics.json files found');  return

    out_path = op_root / 'model_comparison_summary.txt'
    with out_path.open('w') as f:
        sep = '=' * 50
        final_sep = '=' * 80
        for model, text in summaries.items():
            f.write(
                f"{model}\n"
                f"{'-' * 7}\n"
                f"Statistics Summary for op agent (mean ± std):\n"
                f"{sep}\n"
                f"{text}\n"
                f"{sep}\n"
                f"{final_sep}\n\n"
            )
    print(f"Model comparison summary written to {out_path}")

if __name__ == '__main__':
    main()
