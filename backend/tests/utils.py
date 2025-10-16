from datetime import datetime
from itertools import zip_longest
from codebleu import calc_codebleu
import Levenshtein
import tokenize
from io import StringIO
import numpy as np

def calculate_mape(actual, predicted, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between two arrays.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE as a decimal (0-1 range)
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Avoid division by zero by using maximum of actual value or epsilon
    mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), epsilon)))
    
    return mape

def strip_comments(code):
    """Remove all comments from Python code while preserving line structure."""
    result = []
    lines = StringIO(code).readlines()
    
    for line in lines:
        # Remove inline comments and trailing whitespace
        code_line = line.split('#')[0].rstrip()
        # Only add non-empty lines
        if code_line.strip():
            result.append(code_line)
    
    return '\n'.join(result)

def print_comparison(ref, pred, normalized_levenshtein=None, should_calc_codebleu=True):
    # Strip comments before comparison
    ref_clean = strip_comments(ref)
    pred_clean = strip_comments(pred)
    # Calculate metrics using cleaned code
    levenshtein_distance = Levenshtein.distance(ref_clean, pred_clean)
    
    # Show original code with comments for visual comparison
    compare_line_by_line(ref, pred)
    print(f"Levenshtein Distance: {levenshtein_distance}")
    if normalized_levenshtein is not None:
        print(f"Normalized Levenshtein Distance: {normalized_levenshtein:.4f}")
    if should_calc_codebleu:
        code_bleu = calc_codebleu([ref_clean], [pred_clean], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                                  tokenizer=None)
        print(code_bleu)
    print("-" * 50)
    print()

def compare_line_by_line(ref, pred):
    # Split the reference and prediction text by lines
    ref_lines = ref.split("\n")
    pred_lines = pred.split("\n")

    # Calculate the maximum width for each column to align consistently
    ref_width = max(len(line) for line in ref_lines) + 5
    pred_width = max(len(line) for line in pred_lines) + 5

    # Print header with pred and ref names, aligned with calculated widths
    print("Reference".ljust(ref_width + 3) + "| Prediction")
    print("-" * (ref_width + pred_width + 3))  # Separator line

    # Print each line, left-justified for reference and prediction
    for i, (r, p) in enumerate(zip_longest(ref_lines, pred_lines, fillvalue="")):
        print(f"{i}: {r.ljust(ref_width)}| {p}")

def print_log_comparison(gt_logs, pred_logs, command=None, match_func=None):
    """Print a side-by-side comparison of ground truth and predicted logs."""
    # Print the command/prompt if provided
    if command:
        print("\nCOMMAND:")
        print("-" * 80)
        print(command)
        print("-" * 80)

    print("\n" + "="*100)
    print(" "*35 + "LOGS COMPARISON")
    print("="*100)

    # Determine the maximum length for formatting (ignoring blank lines)
    combined = [l for l in gt_logs + pred_logs if l]
    max_pv_len  = max((len(str(l[0])) for l in combined), default=10)
    max_val_len = max((len(str(l[2])) for l in combined), default=10)

    # Calculate column widths
    col_width = max(max_pv_len + max_val_len + 25, 45)  # Minimum width of 45 chars per column

    # Print headers
    header_format = f"| {{:<{max_pv_len}}} | {{:<16}} | {{:<{max_val_len}}} |"
    side_by_side_header = f"{'GROUND TRUTH':^{col_width}} | {'PREDICTED':^{col_width}}"
    divider = "-" * (col_width * 2 + 3)

    print(divider)
    print(side_by_side_header)
    print(divider)

    # Prepare both log lists for side-by-side display
    max_rows = max(len(gt_logs), len(pred_logs))

    matches = 0
    mismatches = 0

    for i in range(max_rows):
        gt_entry   = gt_logs[i]   if i < len(gt_logs)   else None
        pred_entry = pred_logs[i] if i < len(pred_logs) else None

        # Format ground truth log
        if gt_entry:
            pv, ts, val = gt_entry
            time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]
            gt_formatted = header_format.format(str(pv), time_str, str(val))
        else:
            gt_formatted = " " * col_width

        # Format predicted log
        if pred_entry:
            pv, ts, val = pred_entry
            time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]
            pred_formatted = header_format.format(str(pv), time_str, str(val))
        else:
            pred_formatted = " " * col_width

        # Match status using optional matcher
        if gt_entry and pred_entry:                 # both present
            if match_func:
                matched = match_func(gt_entry, pred_entry)
            else:
                gt_pv, _, gt_val = gt_entry
                pred_pv, _, pred_val = pred_entry

                if gt_pv == pred_pv:
                    try:
                        numeric_match = abs(float(gt_val) - float(pred_val)) < 0.001
                        matched = numeric_match
                    except (ValueError, TypeError):
                        string_match = str(gt_val) == str(pred_val)
                        matched = string_match
                else:
                    matched = False
            match_status = "✓" if matched else "✗"
            if matched:
                matches += 1
            else:
                mismatches += 1
        elif gt_entry or pred_entry:                # only one side present
            match_status = "✗"
            mismatches += 1

        # Print side by side with match status
        print(f"{gt_formatted:{col_width}} | {pred_formatted:{col_width}} {match_status}")

    print(divider)

    # Find missing or extra PVs
    gt_pvs   = [log[0] for log in gt_logs  if log]   # skip Nones
    pred_pvs = [log[0] for log in pred_logs if log]

    missing_pvs = set(gt_pvs) - set(pred_pvs)
    extra_pvs = set(pred_pvs) - set(gt_pvs)

    if missing_pvs:
        print("\nMISSING PVs in prediction (present in ground truth):")
        for pv in missing_pvs:
            print(f"  - {pv}")

    if extra_pvs:
        print("\nEXTRA PVs in prediction (not in ground truth):")
        for pv in extra_pvs:
            print(f"  - {pv}")

    # Number of real entries (ignoring None)
    gt_count   = sum(1 for l in gt_logs  if l)
    pred_count = sum(1 for l in pred_logs if l)

    print("\nSUMMARY:")
    print(f"  Ground Truth: {gt_count} log entries")
    print(f"  Predicted:    {pred_count} log entries")
    print(f"  Matches:      {matches}")
    print(f"  Mismatches:   {mismatches}")
    print(f"  Difference:   {abs(gt_count - pred_count)} entries")
    print("="*100 + "\n")