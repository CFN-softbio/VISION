#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROMPT_KEY_RE = re.compile(r'(?:^|_)prompts?(?:$|_)')


def _gather_strings(obj: Any) -> list[str]:
    """Collect all string leaves from a nested structure."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_gather_strings(v))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            out.extend(_gather_strings(v))
    return out


def _collect_prompts_from_json(obj: Any) -> list[str]:
    """
    Collect prompt-like strings from a parsed JSON object.
    - Any value under keys containing 'prompt'
    - Messages[*].content (and nested text parts)
    """
    prompts: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if PROMPT_KEY_RE.search(kl):
                    prompts.extend(_gather_strings(v))
                elif kl == "messages" and isinstance(v, list):
                    for msg in v:
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str):
                                prompts.append(content)
                            elif isinstance(content, list):
                                # content may be structured (e.g., [{"type":"text","text":"..."}])
                                for part in content:
                                    if isinstance(part, dict):
                                        t = part.get("text")
                                        if isinstance(t, str):
                                            prompts.append(t)
                else:
                    walk(v)
        elif isinstance(o, list) or isinstance(o, tuple):
            for item in o:
                walk(item)

    walk(obj)
    return prompts


def _collect_prompts_in_run(run_dir: Path) -> list[str]:
    """Search a run directory for prompt strings."""
    prompts: list[str] = []

    # JSON files (including ndjson)
    for jp in list(run_dir.rglob("*.json")):
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            prompts.extend(_collect_prompts_from_json(data))
        except Exception:
            # Ignore unreadable or invalid json files
            continue

    for nj in list(run_dir.rglob("*.ndjson")):
        try:
            with nj.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        prompts.extend(_collect_prompts_from_json(data))
                    except Exception:
                        continue
        except Exception:
            continue

    # Text-like files explicitly named with 'prompt'
    for tp in list(run_dir.rglob("*prompt*")):
        if tp.is_file() and tp.suffix.lower() in {".txt", ".md", ".log", ".yaml", ".yml"}:
            try:
                prompts.append(tp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue

    return prompts


def _iter_run_dirs(exp_root: Path) -> Iterable[Path]:
    """Yield run directories under each model dir: <exp_root>/<model>/<run_dir>."""
    if not exp_root.exists():
        return []
    for model_dir in sorted(p for p in exp_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        subdirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
        for run_dir in subdirs:
            yield run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check prompts for a saved experiment name and report run paths that do not contain a given string."
    )
    parser.add_argument("experiment", help="Experiment/run name under saved_named_runs/")
    parser.add_argument("needle", help="Substring to search for within prompts")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Override source root (defaults to tests/results/op_cog next to this script)",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Case-insensitive match (default: case-sensitive)",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Also print run paths that contain the substring",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information (counts, summary)",
    )

    args = parser.parse_args()

    op_root = args.source_root or Path(__file__).resolve().parent
    exp_root = op_root / "saved_named_runs" / args.experiment

    if not exp_root.exists():
        print(f"[error] Experiment path not found: {exp_root}", file=sys.stderr)
        return 2

    needle = args.needle
    if args.case_insensitive:
        needle = needle.lower()

    total_runs = 0
    missing = 0

    for run_dir in _iter_run_dirs(exp_root):
        total_runs += 1
        prompts = _collect_prompts_in_run(run_dir)
        found = False
        for p in prompts:
            hay = p.lower() if args.case_insensitive else p
            if needle in hay:
                found = True
                break

        if found:
            if args.show_ok:
                print(f"[ok] {run_dir}")
        else:
            missing += 1
            print(f"[missing] {run_dir}")

    if args.verbose:
        print(f"Scanned {total_runs} runs under {exp_root}. Missing: {missing}")

    # Return non-zero if any missing
    return 1 if missing > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
