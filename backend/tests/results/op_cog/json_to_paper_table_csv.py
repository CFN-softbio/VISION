import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Tuple, List
from collections import defaultdict

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

def _load_mapping(mapping_path: str | None) -> dict:
    if not mapping_path:
        return {}
    p = Path(mapping_path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {p}")
    with p.open('r', encoding='utf-8') as fh:
        return json.load(fh)

def _get_codebleu_from_comp(comp: dict) -> Tuple[float | None, float | None]:
    abd = comp.get('average_best_codebleu')
    if isinstance(abd, dict) and 'comb_7_CodeBLEU' in abd:
        return _to_mean_std(abd.get('comb_7_CodeBLEU'), scale=100.0)
    return None, None

def _row_from_model(model_key: str, model_data: dict, complexity: str, mapping: dict) -> dict | None:
    # Check if model is archived in the mapping
    meta = mapping.get(model_key, {})
    if meta.get('archived', False):
        return None
    
    # Also exclude if the model_key itself is "archived"
    if model_key.lower() == 'archived':
        return None
    
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

    # Build row (skip if key metrics are missing)
    if fscore_m is None and exact_m is None and func_m is None and lev_m is None and inf_m is None:
        return None

    display = meta.get('display', model_key)
    group = meta.get('group', 'Uncategorized')
    vendor = meta.get('vendor', '')
    thinking = bool(meta.get('thinking', model_data.get('thinking', comp.get('thinking', False))))

    return {
        'group': group,
        'vendor': vendor,
        'model_key': model_key,
        'model': display + (r'\tnote{t}' if thinking else ''),
        'thinking': thinking,
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
    }

def _fmt_pm(m: float | None, s: float | None, *, decimals: int = 1) -> str:
    if m is None and s is None:
        return r'\multicolumn{1}{c}{—}'
    if m is None:
        return r'\multicolumn{1}{c}{—}'
    if s is None:
        return f"{m:.{decimals}f} \\pm 0.00"
    return f"{m:.{decimals}f} \\pm {s:.{decimals}f}"

def write_latex_rows(rows: List[dict], out_path: Path, *, decimals: int = 2, columns: List[str] | None = None) -> None:
    # Determine which metric columns to include (after the Model column)
    default_cols = ['avg_full_score', 'functional_match', 'exact_match', 'norm_lev', 'inference_time']
    cols = columns or default_cols

    # Helper to map a metric token to its mean/std keys in the row dict
    def keys_for(token: str) -> Tuple[str, str]:
        return f'{token}_mean', f'{token}_std'

    num_cols = 1 + len(cols)  # Model + selected metrics

    group_order = {'Closed Source Models': 0, 'Open Source Models': 1, 'Uncategorized': 2}
    groups: dict[str, dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    debug_rows: List[dict] = []
    human_rows: List[dict] = []
    
    for r in rows:
        group_name = r.get('group', '').lower()
        # Separate out "debug" group to force it to the bottom
        if group_name == 'debug':
            debug_rows.append(r)
        # Separate out "human" or "humans" group to force it below debug
        elif group_name in ('human', 'humans'):
            human_rows.append(r)
        else:
            groups[r.get('group', 'Uncategorized')][r.get('vendor', '')].append(r)

    # Determine dynamic group ordering by the highest avg_full_score_mean within each group
    group_best: dict[str, float] = {}
    for g_name, vendor_map in groups.items():
        best = 0.0
        for vendor_rows in vendor_map.values():
            for r in vendor_rows:
                v = r.get('avg_full_score_mean') or 0.0
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
                if fv > best:
                    best = fv
        group_best[g_name] = best

    lines: List[str] = []
    for g in sorted(groups.keys(), key=lambda k: (-group_best.get(k, 0.0), group_order.get(k, 99), k)):
        # Group heading with rules above and below
        lines.append(r'\midrule')
        lines.append(rf'\multicolumn{{{num_cols}}}{{l}}{{\textbf{{{g}}}}} \\')
        lines.append(r'\midrule')

        # Order vendors within the group by their best avg_full_score_mean (desc), then by name
        vendor_best: dict[str, float] = {}
        for v_name, v_rows in groups[g].items():
            best = 0.0
            for rr in v_rows:
                val = rr.get('avg_full_score_mean') or 0.0
                try:
                    fv = float(val)
                except Exception:
                    fv = 0.0
                if fv > best:
                    best = fv
            vendor_best[v_name] = best

        first_vendor = True
        for v in sorted(groups[g].keys(), key=lambda vn: (-vendor_best.get(vn, 0.0), vn)):
            vendor_rows = groups[g][v]
            vendor_rows.sort(key=lambda r: (r.get('avg_full_score_mean') or 0.0), reverse=True)

            if g == 'Closed Source Models' and v:
                # Add a rule before vendor headings, except for the first vendor in the group
                if not first_vendor:
                    lines.append(r'\midrule')
                lines.append(rf'\multicolumn{{{num_cols}}}{{l}}{{\textit{{{v}}}}} \\')
                first_vendor = False

            for r in vendor_rows:
                model = r['model']
                if r.get('thinking', False) and r'\tnote{t}' not in model:
                    model = model + r'\tnote{t}'
                metric_parts = []
                for tok in cols:
                    m_key, s_key = keys_for(tok)
                    metric_parts.append(_fmt_pm(r.get(m_key), r.get(s_key), decimals=decimals))
                line = '  ' + ' & '.join([model] + metric_parts) + r' \\'
                lines.append(line)

    # Add debug rows at the bottom if any exist
    if debug_rows:
        lines.append(r'\midrule')
        for r in debug_rows:
            model = r['model']
            if r.get('thinking', False) and r'\tnote{t}' not in model:
                model = model + r'\tnote{t}'
            metric_parts = []
            for tok in cols:
                m_key, s_key = keys_for(tok)
                # Force inference_time to be a dash for debug rows
                if tok == 'inference_time':
                    metric_parts.append(r'\multicolumn{1}{c}{—}')
                else:
                    metric_parts.append(_fmt_pm(r.get(m_key), r.get(s_key), decimals=decimals))
            line = '  ' + ' & '.join([model] + metric_parts) + r' \\'
            lines.append(line)

    # Add human rows at the very bottom if any exist
    if human_rows:
        lines.append(r'\midrule')
        lines.append(rf'\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Humans}}}} \\')
        lines.append(r'\midrule')
        for r in human_rows:
            model = r['model']
            if r.get('thinking', False) and r'\tnote{t}' not in model:
                model = model + r'\tnote{t}'
            metric_parts = []
            for tok in cols:
                m_key, s_key = keys_for(tok)
                # Force inference_time to be a dash for human rows
                if tok == 'inference_time':
                    metric_parts.append(r'\multicolumn{1}{c}{—}')
                else:
                    metric_parts.append(_fmt_pm(r.get(m_key), r.get(s_key), decimals=decimals))
            line = '  ' + ' & '.join([model] + metric_parts) + r' \\'
            lines.append(line)

    Path(out_path).write_text('\n'.join(lines) + '\n\\bottomrule', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Convert model JSON to CSV for paper tables.")
    parser.add_argument('input', help="Path to model_comparison_summary.json (or the .txt; auto-converts).")
    parser.add_argument('--output', '-o', help="Output CSV path. Defaults to <input>.simple.csv (or .json->.csv).")
    parser.add_argument('--complexity', default='simple', choices=['simple', 'complex'],
                        help="Which complexity subset to export.")
    parser.add_argument('--mapping', help="Optional JSON mapping: {model_key: {display, group, vendor}}.")
    parser.add_argument('--drop-missing', action='store_true',
                        help="Drop rows with any missing required metric.")
    parser.add_argument('--sort-by', default='avg_full_score_mean',
                        choices=['avg_full_score_mean','exact_match_mean','functional_match_mean','norm_lev_mean','inference_time_mean'],
                        help="Primary sort column.")
    parser.add_argument('--desc', action='store_true', help="Sort descending.")
    parser.add_argument('--latex-rows', help="Optional path to write LaTeX table rows (to \\input in your tabular).")
    parser.add_argument('--latex-columns', default=None,
                        help="Comma-separated metric columns for LaTeX rows. Valid tokens: avg_full_score, exact_match, functional_match, comb_7_CodeBLEU, norm_lev, inference_time. Default (simple): avg_full_score,functional_match,exact_match,norm_lev,inference_time; Default (complex): avg_full_score,functional_match,comb_7_CodeBLEU,norm_lev,inference_time.")
    parser.add_argument('--decimals', type=int, default=1, help="Decimal places for mean/std in LaTeX rows.")
    args = parser.parse_args()

    in_path = Path(args.input)
    data = _load_json_from_input(in_path)
    mapping = _load_mapping(args.mapping)

    rows = []
    for model_key, model_data in data.items():
        row = _row_from_model(model_key, model_data, args.complexity, mapping)
        if row is None:
            continue
        if getattr(args, 'drop_missing', False) and any(row[k] is None for k in (
            'avg_full_score_mean','exact_match_mean','functional_match_mean','norm_lev_mean','inference_time_mean'
        )):
            continue
        rows.append(row)

    # Sort (stable sort by group then requested metric)
    rows.sort(key=lambda r: (r.get('group',''),), reverse=False)
    rows.sort(key=lambda r: (r.get(args.sort_by),), reverse=args.desc)

    # Output path
    if args.output:
        out_csv = Path(args.output)
    else:
        base = in_path.with_suffix('')
        if in_path.suffix.lower() == '.txt':
            out_csv = base.with_suffix(f'.{args.complexity}.csv')
        else:
            out_csv = in_path.with_suffix(f'.{args.complexity}.csv')

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'group','vendor','model_key','model',
        'avg_full_score_mean','avg_full_score_std',
        'functional_match_mean','functional_match_std',
        'exact_match_mean','exact_match_std',
        'comb_7_CodeBLEU_mean','comb_7_CodeBLEU_std',
        'norm_lev_mean','norm_lev_std',
        'inference_time_mean','inference_time_std',
    ]
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if args.latex_rows:
        valid_tokens = {'avg_full_score', 'exact_match', 'functional_match', 'comb_7_CodeBLEU', 'norm_lev', 'inference_time'}
        if args.latex_columns:
            col_tokens = [t.strip() for t in args.latex_columns.split(',') if t.strip()]
        else:
            if args.complexity == 'complex':
                col_tokens = ['avg_full_score', 'functional_match', 'comb_7_CodeBLEU', 'norm_lev', 'inference_time']
            else:
                col_tokens = ['avg_full_score', 'functional_match', 'exact_match', 'norm_lev', 'inference_time']
        invalid = [t for t in col_tokens if t not in valid_tokens]
        if invalid:
            raise ValueError(f"Invalid --latex-columns tokens: {invalid}. Valid: {sorted(valid_tokens)}")
        write_latex_rows(rows, Path(args.latex_rows), decimals=args.decimals, columns=col_tokens)
        print(f"Wrote LaTeX rows: {args.latex_rows}")

    print(f"Wrote CSV: {out_csv}")

if __name__ == '__main__':
    main()
