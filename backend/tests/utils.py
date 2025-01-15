from itertools import zip_longest
from codebleu import calc_codebleu
import Levenshtein
import tokenize
from io import StringIO

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
