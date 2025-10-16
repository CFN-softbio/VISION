"""
Print a detailed score breakdown (timing, temperature, full) for entries
contained in an Op-cog CSV result file.

Usage
-----
python -m tests.score_breakdown  path/to/results_op_agent.csv  -n 10
"""
from __future__ import annotations
import argparse, csv, json, sys, types, importlib
from tests.test_op_cog import OpAgentTestFramework
from utils import print_log_comparison

# ----------------------------------------------------------------------
# Light stubs for heavy / site-specific modules that
# tests.test_op_cog imports transitively.  They are **not**
# used by this utility – we just provide minimal dummies so the
# import succeeds without requiring a full EPICS / caching setup.
# ----------------------------------------------------------------------
for _m in ("epics", "pexpect",
           "codebleu", "sklearn", "sklearn.metrics"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# provide a stub execution_cache only if the real module is missing
try:
    import execution_cache        # noqa: F401
except ModuleNotFoundError:
    sys.modules["execution_cache"] = types.ModuleType("execution_cache")

# minimal API surface that test_op_cog expects
_ep = sys.modules["epics"]
for _f in ("caget", "caput", "caput_many", "camonitor", "camonitor_clear"):
    setattr(_ep, _f, lambda *a, **k: None)

sys.modules["codebleu"].calc_codebleu = lambda *a, **k: {}
sys.modules["sklearn.metrics"].r2_score = lambda *a, **k: 0.0

# make tests.utils visible as top-level “utils” so _timing_match works
sys.modules["utils"] = importlib.import_module("tests.utils")

TEMP_PV = "XF:11BM-ES:{LINKAM}:TEMP"

def _log_entries_match(gt, pred):
    """
    Decide whether two PV-log entries match.
    Each entry is (pv_name, timestamp, value) or None.
    """
    if gt is None or pred is None:
        return False
    gt_pv, _, gt_val = gt
    pr_pv, _, pr_val = pred
    if gt_pv != pr_pv:
        return False
    try:
        return abs(float(gt_val) - float(pr_val)) < 1e-3
    except (ValueError, TypeError):
        return str(gt_val) == str(pr_val)

# ------------------------------------------------------------------ #
def _print_entry(row: dict[str, str], timing_det: dict, temp_det: dict,
                 gt_snippets: list[str], temperature_involved: bool,
                 gt_temp_logs, pred_temp_logs) -> None:
    print("\n" + "=" * 120)
    print("COMMAND:", row["command"])
    print("-" * 120)
    print("GROUND-TRUTH SNIPPETS:")
    for idx, snip in enumerate(gt_snippets, 1):
        print(f"--- [GT {idx}] " + "-" * 100)
        print(snip)
    print("-" * 120)
    print("PREDICTED SNIPPET:")
    print(row["generated_code"])
    print("-" * 120)
    print("TIMING DETAILS:", timing_det)
    print("TEMPERATURE DETAILS:", temp_det)
    print("FULL SCORE (from CSV):", row["full_score"])
    # ----- continuous-score breakdown ---------------------------------
    pv_rate      = float(row.get("pv_match_rate", 0))
    timing_score = float(row.get("timing_score", 0))
    temp_score   = float(row.get("temp_score", 0))

    if temperature_involved:
        contribs = [
            ("PV match rate", pv_rate,      0.6),
            ("Timing score",  timing_score, 0.2),
            ("Temp score",    temp_score,   0.2),
        ]
    else:
        contribs = [
            ("PV match rate", pv_rate,      0.8),
            ("Timing score",  timing_score, 0.2),
        ]

    print("FULL-SCORE BREAKDOWN:")
    subtotal = 0.0
    for name, val, w in contribs:
        part = val * w
        subtotal += part
        print(f"  {name:15s}: {val:6.3f} × {w:3.1f} = {part:6.3f}")
    csv_full = float(row.get("full_score", 0))
    print(f"  {'-'*38}")
    print(f"  Calculated total : {subtotal:6.3f}")
    print(f"  Stored in CSV    : {csv_full:6.3f}")
    print(f"  Δ(calc-CSV)      : {subtotal - csv_full:+.3e}")
    print("-" * 120)
    print("=" * 120 + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description="Show score components stored "
                                             "in an Op-cog result CSV.")
    ap.add_argument("csv", help="results_op_agent.csv file")
    ap.add_argument("-n", "--num", type=int, default=0,
                    help="number of rows to display, default: 0 (all)")
    args = ap.parse_args()

    shown = 0
    with open(args.csv, encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # decode JSON columns ------------------------------------------------
            gt_logs   = json.loads(row["gt_pv_logs"]   or "[]")
            pr_logs   = json.loads(row["pred_pv_logs"] or "[]")
            ignore    = set(json.loads(row["ignore_pvs"] or "[]"))
            exp_codes = json.loads(row["expected_codes"])
            aligned_gt = json.loads(row.get("aligned_gt_logs")  or "[]")
            aligned_pr = json.loads(row.get("aligned_pred_logs") or "[]")
            ground_truth_snippets = exp_codes      # use all variants verbatim

            # timing component ---------------------------------------------------
            gt_ts = [ts for pv, ts, _ in gt_logs
                     if pv not in ignore and pv != TEMP_PV]
            pr_ts = [ts for pv, ts, _ in pr_logs
                     if pv not in ignore and pv != TEMP_PV]
            _, _, timing_det = OpAgentTestFramework._timing_match(None, gt_ts, pr_ts)

            # NEW: use gt_temp_logs and pred_temp_logs directly from CSV
            gt_temp_logs   = json.loads(row.get("gt_temp_logs") or "[]")
            pred_temp_logs = json.loads(row.get("pred_temp_logs") or "[]")
            temperature_involved = bool(gt_temp_logs) or bool(pred_temp_logs)

            # Prefer whatever was stored in the CSV
            raw_temp = row.get("temp_score", "")
            try:
                parsed = json.loads(raw_temp)               # works for “0.93” and for JSON dicts
            except (TypeError, json.JSONDecodeError):
                parsed = raw_temp

            if isinstance(parsed, dict):
                temp_det = parsed                           # full dict already present
            else:
                try:
                    temp_det = {"score": float(parsed)}     # simple numeric score
                except (ValueError, TypeError):
                    temp_det = {"score": None}              # fallback / missing

            # Always use the snippet chosen by the original test run
            # TODO: get ground_truth_snippet by re-calculating which candidate was the best

            _print_entry(row, timing_det, temp_det,
                         ground_truth_snippets, temperature_involved,
                         gt_temp_logs, pred_temp_logs)

            print_log_comparison(
                aligned_gt,
                aligned_pr,
                row["command"],
                match_func=_log_entries_match,
            )

            shown += 1
            if shown >= args.num > 0:
                break

if __name__ == "__main__":
    main()
