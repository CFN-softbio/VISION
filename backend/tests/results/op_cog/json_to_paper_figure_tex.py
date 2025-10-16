import argparse
import json
import re
import ast
from pathlib import Path
from typing import Any, Tuple, List, Optional

CSV_SEP = ';'

#
# Helpers adapted from json_to_paper_table_csv.py
#

def _to_mean_std(val: Any, *, scale: float = 1.0) -> Tuple[float | None, float | None]:
    if val is None:
        return None, None
    if isinstance(val, (int, float)):
        return float(val) * scale, None
    if isinstance(val, dict):
        mean = val.get('mean', None)
        std = val.get('std', None)
        if mean is None:
            return None, None
        m = float(mean) * scale
        s = float(std) * scale if isinstance(std, (int, float)) else None
        return m, s
    if isinstance(val, str):
        m = re.match(r'^\s*([0-9.]+)\s*±\s*([0-9.]+)\s*$', val)
        if m:
            return float(m.group(1)) * scale, float(m.group(2)) * scale
        try:
            return float(val) * scale, None
        except ValueError:
            return None, None
    return None, None

def _load_json_from_input(input_path: Path) -> dict:
    if input_path.suffix.lower() == '.json':
        with input_path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    if input_path.suffix.lower() == '.txt':
        # Auto-convert to JSON using the existing converter
        from tests.results.op_cog.convert_model_summary_txt_to_json import convert_model_summary_txt_to_json
        out_json = convert_model_summary_txt_to_json(input_path)
        with out_json.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported input extension: {input_path.suffix}")

def _load_mapping(mapping_path: str | None, input_path: Path | None = None) -> dict:
    """
    Load mapping JSON. If mapping_path is None, tries:
    - <input_path>/model_table_mapping.json
    - <this_script_dir>/model_table_mapping.json
    """
    candidates: List[Path] = []
    if mapping_path:
        candidates.append(Path(mapping_path))
    if input_path is not None:
        candidates.append(input_path.with_name('model_table_mapping.json'))
    try:
        candidates.append(Path(__file__).with_name('model_table_mapping.json'))
    except NameError:
        pass

    seen: set[str] = set()
    for p in candidates:
        ps = str(p.resolve()) if p else ""
        if not ps or ps in seen:
            continue
        seen.add(ps)
        if p.exists():
            with p.open('r', encoding='utf-8') as fh:
                return json.load(fh)

    if mapping_path:
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    return {}

def _get_codebleu_from_comp(comp: dict) -> Tuple[float | None, float | None]:
    abd = comp.get('average_best_codebleu')
    if isinstance(abd, dict) and 'comb_7_CodeBLEU' in abd:
        return _to_mean_std(abd.get('comb_7_CodeBLEU'), scale=100.0)
    return None, None

def _extract_component_metric(comp: dict, candidates: List[str], scale: float = 100.0) -> Tuple[float | None, float | None]:
    """
    Try to extract a component metric (e.g., PV match, timing, temp) from a variety of possible keys
    or nested containers. Returns (mean, std) scaled to 0-100 if applicable; (None, None) if absent.
    """
    # Try top-level candidates and their *_mean variants
    for key in (candidates + [f"{c}_mean" for c in candidates]):
        if key in comp:
            return _to_mean_std(comp.get(key), scale=scale)

    # Try common nested containers
    for container in ("components", "component_metrics", "metrics_by_component", "component_scores"):
        nested = comp.get(container)
        if isinstance(nested, dict):
            for base in candidates:
                variants = {
                    base,
                    base.replace("_", " "),
                    base.replace("_", "-"),
                    base.replace("_", " ").title(),
                    base.replace("_", "-").title(),
                    base.upper(),
                    base.lower(),
                    f"{base}_mean",
                }
                for v in variants:
                    if v in nested:
                        return _to_mean_std(nested.get(v), scale=scale)

    return None, None

def _row_from_model(model_key: str, model_data: dict, complexity: str, mapping: dict) -> dict | None:
    comp = model_data.get('metrics_by_complexity', {}).get(complexity, {})
    if not isinstance(comp, dict) or not comp:
        return None

    # Extract required metrics
    fscore_m, fscore_s = _to_mean_std(comp.get('average_full_score'), scale=100.0)
    exact_m, exact_s = _to_mean_std(comp.get('exact_code_match_rate'))
    func_m, func_s = _to_mean_std(comp.get('accuracy'))
    lev_m, lev_s = _to_mean_std(comp.get('average_best_normalized_levenshtein'), scale=100.0)
    inf_m, inf_s = _to_mean_std(comp.get('average_inference_time'))
    codebleu_m, codebleu_s = _get_codebleu_from_comp(comp)

    # Additional component metrics (best-effort; optional)
    pv_m, pv_s = _extract_component_metric(comp, ["average_pv_match_rate", "pv_match", "pvmatch", "pv", "process_variable_match"], scale=1.0)
    tm_m, tm_s = _extract_component_metric(comp, ["average_timing_score", "timing", "timing_score", "time", "timing_match"], scale=100.0)
    tp_m, tp_s = _extract_component_metric(comp, ["average_temp_score", "temp", "temperature", "temperature_score"], scale=100.0)

    # Build row (skip if key metrics are missing)
    if fscore_m is None and exact_m is None and func_m is None and lev_m is None and inf_m is None:
        return None

    meta = _find_mapping_meta(mapping, model_key)
    display = meta.get('display', model_key)
    group = meta.get('group', 'Uncategorized')
    vendor = meta.get('vendor', '')

    return {
        'group': group,
        'vendor': vendor,
        'model_key': model_key,
        'model': display,
        'avg_full_score_mean': fscore_m,
        'avg_full_score_std': fscore_s,
        'exact_match_mean': exact_m,
        'exact_match_std': exact_s,
        'functional_match_mean': func_m,
        'functional_match_std': func_s,
        'comb_7_CodeBLEU_mean': codebleu_m,
        'comb_7_CodeBLEU_std': codebleu_s,
        'norm_lev_mean': lev_m,
        'norm_lev_std': lev_s,
        'inference_time_mean': inf_m,
        'inference_time_std': inf_s,
        # Optional component metrics for specialized figures
        'pv_match_mean': pv_m,
        'pv_match_std': pv_s,
        'timing_mean': tm_m,
        'timing_std': tm_s,
        'temp_mean': tp_m,
        'temp_std': tp_s,
    }

#
# Figure generator
#

DEFAULT_FIGURES = {
    # Simple-flow stacked figure:
    # left bar: fullscore; right bar: stack exact (top) + improvement (bottom) where improvement = functional - exact
    "simple_flow": {
        "columns": [
            {"name": "label", "expr": "display"},
            {"name": "fullscore", "source": "avg_full_score_mean"},
            {"name": "exact", "source": "exact_match_mean"},
            {"name": "improvement", "expr": "functional_match_mean - exact_match_mean"},
            {"name": "total", "source": "functional_match_mean"},
        ],
        "sort_by_row_key": "avg_full_score_mean",
        "desc": True,
        "comment": "SIMPLE-flow data (CSV; safer parsing)"
    },
    # Complex-flow stacked figure:
    # left bar: fullscore; right bar: stack exact (top) + improvement (bottom) where improvement = functional - exact
    "complex_flow": {
        "columns": [
            {"name": "label", "expr": "display"},
            {"name": "fullscore", "source": "avg_full_score_mean"},
            {"name": "exact", "source": "exact_match_mean"},
            {"name": "improvement", "expr": "functional_match_mean - exact_match_mean"},
            {"name": "total", "source": "functional_match_mean"},
        ],
        "sort_by_row_key": "avg_full_score_mean",
        "desc": True,
        "comment": "COMPLEX-flow data (CSV; safer parsing)"
    },
    # Complex components figure (bar groups for fullscore and component metrics):
    # columns: model key (for referencing), label (human-friendly), fullscore, pvmatch, timing, temp
    "complex_components": {
        "columns": [
            {"name": "model", "text": "model_key"},
            {"name": "label", "expr": "display"},
            {"name": "fullscore", "source": "avg_full_score_mean"},
            {"name": "pvmatch", "source": "pv_match_mean"},
            {"name": "timing", "source": "timing_mean"},
            {"name": "temp", "source": "temp_mean"},
        ],
        "sort_by_row_key": "avg_full_score_mean",
        "desc": True,
        "csv_sep": ",",
        "fill_missing_zero": True,
        "comment": "COMPLEX components data (comma-separated CSV)"
    },
    # Simple-flow (old dataset + old prompt) figure:
    # Same columns as simple_flow, but emits comma-separated CSV to match legacy figures.
    "simple_flow_old": {
        "columns": [
            {"name": "label", "expr": "display"},
            {"name": "fullscore", "source": "avg_full_score_mean"},
            {"name": "exact", "source": "exact_match_mean"},
            {"name": "improvement", "expr": "functional_match_mean - exact_match_mean"},
            {"name": "total", "source": "functional_match_mean"},
        ],
        "sort_by_row_key": "avg_full_score_mean",
        "desc": True,
        "csv_sep": ",",
        "comment": "SIMPLE-flow (old dataset + old prompt) data (comma-separated CSV)"
    },
}

LATEX_SPECIALS = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

def latex_escape(s: str) -> str:
    return ''.join(LATEX_SPECIALS.get(ch, ch) for ch in s)

def _norm_token(s: str | None) -> str:
    """
    Normalize a token for fuzzy matching: lowercase, apply common synonyms, and strip non-alphanumeric.
    Synonyms: S4 -> Sonnet4, O4 -> Opus4.
    """
    if s is None:
        return ""
    txt = str(s).lower()
    # common short forms used in labels
    txt = re.sub(r'\bs4\b', 'sonnet4', txt)
    txt = re.sub(r'\bo4\b', 'opus4', txt)
    return re.sub(r'[^a-z0-9]+', '', txt)

def _find_mapping_meta(mapping: dict, model_key: str) -> dict:
    """
    Strictly return metadata only for exact mapping key matches.
    No fuzzy matching is performed.
    """
    meta = mapping.get(model_key)
    return meta if isinstance(meta, dict) else {}

def _safe_eval_numeric_expr(expr: str, variables: dict[str, Any]) -> float | None:
    """
    Safely evaluate a simple arithmetic expression with +,-,*,/, parentheses,
    over numeric variables provided in `variables`. Returns None if any
    variable is missing or None.
    """
    # If the expression is a simple variable that might be string-y (like display), don't eval numerically
    if expr in ("display", "model", "model_key"):
        val = variables.get(expr if expr != "model" else "display")
        return None if isinstance(val, str) else val  # Return None for strings in numeric context

    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return float(n.n)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(n.left)
            right = _eval(n.right)
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                try:
                    return left / right
                except ZeroDivisionError:
                    return None
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            operand = _eval(n.operand)
            if operand is None:
                return None
            return +operand if isinstance(n.op, ast.UAdd) else -operand
        if isinstance(n, ast.Name):
            name = n.id
            if name == "model":
                name = "display"
            if name not in variables:
                return None
            val = variables[name]
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            # Non-numeric in numeric expr
            return None
        if isinstance(n, ast.Call):
            # Disallow function calls
            return None
        if isinstance(n, ast.Attribute) or isinstance(n, ast.Subscript):
            # Disallow attr/subscript
            return None
        # Disallow all others
        return None

    return _eval(node)

def _compute_figure_row(row: dict, columns: List[dict], decimals: int, label_override: Optional[str] = None, fill_missing_zero: bool = False) -> Optional[List[str]]:
    """Given a base metrics row and figure column definitions, compute a CSV row (as strings).
       Returns None if any required numeric column is missing (unless fill_missing_zero=True)."""
    # Provide variable context
    vars_ctx: dict[str, Any] = {
        # identity
        "display": row.get("model"),
        "model": row.get("model"),
        "model_key": row.get("model_key"),
        # numerics
        "avg_full_score_mean": row.get("avg_full_score_mean"),
        "exact_match_mean": row.get("exact_match_mean"),
        "functional_match_mean": row.get("functional_match_mean"),
        "comb_7_CodeBLEU_mean": row.get("comb_7_CodeBLEU_mean"),
        "norm_lev_mean": row.get("norm_lev_mean"),
        "inference_time_mean": row.get("inference_time_mean"),
        # optional components
        "pv_match_mean": row.get("pv_match_mean"),
        "timing_mean": row.get("timing_mean"),
        "temp_mean": row.get("temp_mean"),
    }

    out_cells: List[str] = []
    for col in columns:
        name = col["name"]

        # Generic text column support (e.g., "model" as model_key)
        if "text" in col:
            t = col["text"]
            key = "display" if t in ("display", "model") else "model_key" if t == "model_key" else t
            val = vars_ctx.get(key)
            val = "" if val is None else str(val)
            out_cells.append("{" + latex_escape(val) + "}")
            continue

        if name == "label":
            # label: prefer explicit override; otherwise expr==display/model or source to 'model' display
            label_val = None
            if label_override is not None:
                label_val = label_override
            elif "expr" in col:
                if col["expr"] in ("display", "model", "model_key"):
                    label_val = vars_ctx.get("display") if col["expr"] in ("display", "model") else vars_ctx.get("model_key")
                else:
                    evaluated = _safe_eval_numeric_expr(col["expr"], vars_ctx)
                    label_val = "" if evaluated is None else str(evaluated)
            elif "source" in col:
                src = col["source"]
                v = vars_ctx.get(src)
                label_val = "" if v is None else str(v)
            else:
                label_val = vars_ctx.get("display") or ""
            # Escape for LaTeX, wrap in braces
            out_cells.append("{" + latex_escape(str(label_val)) + "}")
            continue

        # Numeric columns
        if "source" in col:
            v = vars_ctx.get(col["source"])
            if v is None:
                if fill_missing_zero:
                    out_cells.append(f"{0.0:.{decimals}f}")
                    continue
                return None
            out_cells.append(f"{float(v):.{decimals}f}")
        elif "expr" in col:
            val = _safe_eval_numeric_expr(col["expr"], vars_ctx)
            if val is None:
                if fill_missing_zero:
                    out_cells.append(f"{0.0:.{decimals}f}")
                    continue
                return None
            out_cells.append(f"{float(val):.{decimals}f}")
        else:
            # Unsupported specification
            return None

    return out_cells

def _filter_order_rows(rows: List[dict], include_keys: List[str] | None, exclude_keys: List[str] | None,
                       sort_by_row_key: str, desc: bool, limit: int | None,
                       preserve_include_order: bool) -> Tuple[List[dict], List[Optional[str]]]:
    # Build indices
    by_key = {r["model_key"]: r for r in rows}
    by_display = {r["model"]: r for r in rows}

    selected: List[dict] = []
    label_overrides: List[Optional[str]] = []
    if include_keys:
        seen = set()
        missing_exact: List[str] = []
        for tok in include_keys:
            tok = tok.strip()
            r = by_key.get(tok) or by_display.get(tok)
            if r is None:
                missing_exact.append(tok)
                continue
            if r["model_key"] not in seen:
                selected.append(r)
                label_overrides.append(None)  # always use mapping display name for labels
                seen.add(r["model_key"])
        if missing_exact:
            raise ValueError(f"Could not match include-keys exactly to available models: {missing_exact}. Provide exact model keys or exact display names from the mapping.")
    else:
        selected = list(rows)
        label_overrides = [None] * len(selected)

    if exclude_keys:
        exclude_set = set(k.strip() for k in exclude_keys)
        pairs = [(r, o) for r, o in zip(selected, label_overrides)
                 if (r["model_key"] not in exclude_set and r["model"] not in exclude_set)]
        if pairs:
            selected, label_overrides = map(list, zip(*pairs))
        else:
            selected, label_overrides = [], []

    # If include_keys provided and preserve order, do not sort
    if not (include_keys and preserve_include_order):
        pairs = list(zip(selected, label_overrides))
        pairs.sort(key=lambda p: (p[0].get(sort_by_row_key) or 0.0), reverse=desc)
        if pairs:
            selected, label_overrides = map(list, zip(*pairs))
        else:
            selected, label_overrides = [], []

    if limit is not None:
        selected = selected[:limit]
        label_overrides = label_overrides[:limit]

    return selected, label_overrides

def build_pgfplots_table(rows: List[dict], figure_def: dict, decimals: int,
                         include_keys: List[str] | None = None,
                         exclude_keys: List[str] | None = None,
                         sort_by_row_key: str | None = None,
                         desc: bool | None = None,
                         limit: int | None = None,
                         preserve_include_order: bool = True) -> Tuple[str, int, List[str]]:
    """
    Build the LaTeX pgfplots CSV block string and return (text, row_count_kept, labels_in_order).
    """
    columns = figure_def["columns"]
    sort_key = sort_by_row_key or figure_def.get("sort_by_row_key", "avg_full_score_mean")
    descending = figure_def.get("desc", True) if desc is None else desc
    sep = figure_def.get("csv_sep", CSV_SEP)

    # Filter/order base rows
    base_rows, label_overrides = _filter_order_rows(rows, include_keys, exclude_keys, sort_key, descending, limit, preserve_include_order)

    # Header
    header = sep.join(col["name"] for col in columns)
    lines = [header]

    kept = 0
    for r, label_override in zip(base_rows, label_overrides):
        cells = _compute_figure_row(r, columns, decimals, label_override=label_override, fill_missing_zero=figure_def.get("fill_missing_zero", False))
        if cells is None:
            continue
        lines.append(sep.join(cells))
        kept += 1

    coords_labels = [latex_escape(lo if lo is not None else (r.get("model") or "")) for r, lo in zip(base_rows, label_overrides)]
    return "\n".join(lines), kept, coords_labels

def main():
    parser = argparse.ArgumentParser(description="Convert model JSON to LaTeX pgfplots CSV blocks for named figures.")
    parser.add_argument('input', help="Path to model_comparison_summary.json (or the .txt; auto-converts).")
    parser.add_argument('--output', '-o', help="Output .tex path. Defaults to <input>.<figure>.<complexity>.tex")
    parser.add_argument('--complexity', default='simple', choices=['simple', 'complex'], help="Which complexity subset to export.")
    parser.add_argument('--mapping', help="Optional JSON mapping: {model_key: {display, group, vendor}}. If omitted, will search for model_table_mapping.json alongside the input or this script.")
    parser.add_argument('--figure', default='simple_flow', choices=list(DEFAULT_FIGURES.keys()),
                        help="Which named figure schema to use.")
    parser.add_argument('--include-keys', nargs='*', help="EXACT model keys or EXACT display names from the mapping to include (pass each as a separate argument; quote values with spaces/commas). If omitted or provided with no values, all models are included.")
    parser.add_argument('--exclude-keys', nargs='+', help="Model keys or display names to exclude (pass each as a separate argument; quote values with spaces/commas).")
    parser.add_argument('--limit', type=int, help="Keep only the first N models after filtering/sorting.")
    parser.add_argument('--sort-by', default=None, help="Sort by this base-row key (e.g., avg_full_score_mean). Defaults per figure.")
    parser.add_argument('--asc', action='store_true', help="Sort ascending (default is descending for most figures).")
    parser.add_argument('--keep-include-order', action='store_true',
                        help="If --include-keys is given, preserve its order (default False).")
    parser.add_argument('--decimals', type=int, default=1, help="Decimal places for numeric values.")
    parser.add_argument('--table-name', default='\\datatable', help="Name of the pgfplots table macro to assign (e.g., \\datatable).")
    args = parser.parse_args()

    in_path = Path(args.input)
    data = _load_json_from_input(in_path)
    mapping = _load_mapping(args.mapping, in_path)

    # Build base metric rows
    rows: List[dict] = []
    for model_key, model_data in data.items():
        r = _row_from_model(model_key, model_data, args.complexity, mapping)
        if r is None:
            continue
        rows.append(r)

    # Parse include/exclude lists
    include_keys = args.include_keys if args.include_keys else None
    exclude_keys = args.exclude_keys if args.exclude_keys else None

    # Enforce exact include-keys against mapping, if provided
    if include_keys is not None:
        if not mapping:
            raise ValueError("When using --include-keys, a mapping file is required. Provide --mapping pointing to model_table_mapping.json.")
        allowed_names = set(mapping.keys())
        for v in mapping.values():
            if isinstance(v, dict):
                disp = v.get('display')
                if isinstance(disp, str):
                    allowed_names.add(disp)
        missing = [tok for tok in include_keys if tok not in allowed_names]
        if missing:
            hint = ""
            if any(',' in tok for tok in missing):
                hint = " Detected a comma in an unmatched item; pass each model as a separate argument (quote items with spaces/commas)."
            raise ValueError(f"--include-keys must match exactly a mapping key or display name: {missing}.{hint}")

    figure_def = DEFAULT_FIGURES[args.figure]
    csv_block, kept, coords_labels = build_pgfplots_table(
        rows,
        figure_def,
        decimals=args.decimals,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        sort_by_row_key=args.sort_by,
        desc=(not args.asc),
        limit=args.limit,
        preserve_include_order=args.keep_include_order
    )

    # Compose LaTeX snippet
    title = f"{args.figure.replace('_',' ').upper()} ({args.complexity})"
    comment = figure_def.get("comment", "pgfplots data (CSV; safer parsing)")
    snippet_lines = []
    snippet_lines.append(f"% --- Data for {title} ---")
    snippet_lines.append(f"% {comment}")
    sep_val = figure_def.get("csv_sep", CSV_SEP)
    sep_token = "comma" if sep_val == "," else "semicolon"
    snippet_lines.append(rf"\pgfplotstableread[col sep={sep_token}]{{")
    snippet_lines.append(csv_block)
    snippet_lines.append(rf"}}{args.table_name}")
    # Also include ordered symbolic x coords as comments
    coords_joined = ",".join("{" + s + "}" for s in coords_labels)
    snippet_lines.append("% symbolic x coords (ordered for this figure):")
    snippet_lines.append(f"% symbolic x coords={{ {coords_joined} }}")

    # Output path
    if args.output:
        out_tex = Path(args.output)
    else:
        out_tex = in_path.with_suffix(f".{args.figure}.{args.complexity}.tex")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(snippet_lines) + "\n", encoding='utf-8')

    print(f"Wrote LaTeX data block with {kept} rows: {out_tex}")
    coords_joined = ",".join("{" + s + "}" for s in coords_labels)
    print("symbolic x coords (ordered):")
    print(f"symbolic x coords={{ {coords_joined} }}")

if __name__ == '__main__':
    main()
