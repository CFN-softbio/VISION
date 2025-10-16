"""
Converts model summary txt files to JSON. Outputs all models with their relevant parameters.

{
  "o3-high": {
    "summary": {
      "accuracy": {
        "mean": 88.4804,
        "std": 1.1232
      },
      "total_entries": {
        "mean": 136.0,
        "std": 0.0
      },
      "correct_matches": {
        "mean": 120.3333,
        "std": 1.5275
      },
      "average_execution_time": {
        "mean": 6.4398,
        "std": 0.3531
      }
    },
    "metrics_by_complexity": {
      "simple": {
        "count": {
          "mean": 116.0,
          "std": 0.0
        },
        ...
      },
      "complex": {
        "count": {
          "mean": 20.0,
          "std": 0.0
        },
        "full_matches": {
          "mean": 9.0,
          "std": 1.7321
        ...
      }
    }
  },
  "llama3.3": {
  ...
  }
}

"""

import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Tuple

def _parse_pm_value(val: str) -> Any:
    """
    Parse 'mean ± std' -> {'mean': float, 'std': float}
    Parse plain numeric -> float
    Otherwise return raw string.
    """
    val = val.strip()
    m = re.match(r'^([0-9]+(?:\.[0-9]+)?)\s*±\s*([0-9]+(?:\.[0-9]+)?)$', val)
    if m:
        return {"mean": float(m.group(1)), "std": float(m.group(2))}
    try:
        return float(val)
    except ValueError:
        return val

def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(' '))

def _parse_nested_metrics(lines: list[str], start_idx: int, base_indent: int) -> Tuple[dict, int]:
    """
    Generic indentation-based parser for the YAML-like block under metrics_by_complexity.
    Returns (parsed_dict, next_index_after_block).
    """
    result: dict = {}
    stack: list[tuple[int, dict]] = [(base_indent - 1, result)]
    i = start_idx

    while i < len(lines):
        raw = lines[i].rstrip('\n')
        if not raw.strip():
            i += 1
            continue
        if re.match(r'^[=-]{5,}', raw):
            break

        indent = _indent_width(raw)
        if indent < base_indent:
            break

        # ascend to the correct parent for this indent
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else result

        stripped = raw.strip()
        if ':' not in stripped:
            i += 1
            continue

        key, rest = stripped.split(':', 1)
        key = key.strip()
        val = rest.strip()

        if val == '':
            parent[key] = {}
            stack.append((indent, parent[key]))
            i += 1
            continue

        parent[key] = _parse_pm_value(val)
        i += 1

    return result, i

def _parse_model_section_to_dict(section_content: str) -> dict:
    """
    Parse one model's section into a dict:
      {
        'summary': { metric: {'mean':..., 'std':...} | float },
        'metrics_by_complexity': {
            'simple': {...},
            'complex': {...}
        }
      }
    Also adds derived improvements (pp and relative) for each complexity if possible.
    """
    lines = section_content.splitlines()
    data: dict = {"summary": {}, "metrics_by_complexity": {}}

    # Find summary block start
    i = 0
    while i < len(lines) and 'Statistics Summary for op agent' not in lines[i]:
        i += 1
    if i < len(lines):
        i += 1
        # skip any '====' lines
        while i < len(lines) and re.match(r'^[=]{5,}', lines[i]):
            i += 1
        # collect summary metrics until 'metrics_by_complexity:'
        while i < len(lines):
            t = lines[i].strip()
            if not t or re.match(r'^[=-]{5,}', t):
                i += 1
                continue
            if t.startswith('metrics_by_complexity:'):
                break
            if ':' in t:
                k, rest = t.split(':', 1)
                data['summary'][k.strip()] = _parse_pm_value(rest.strip())
            i += 1

    # Parse metrics_by_complexity
    j = 0
    while j < len(lines) and not lines[j].strip().startswith('metrics_by_complexity:'):
        j += 1
    if j < len(lines):
        # next non-empty line determines base indent
        k = j + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k < len(lines):
            base_indent = _indent_width(lines[k])
            mbc, _ = _parse_nested_metrics(lines, k, base_indent)
            data['metrics_by_complexity'] = mbc

    # Add derived improvements per complexity if accuracy and exact_code_match_rate present
    for comp in ('simple', 'complex'):
        comp_dict = data.get('metrics_by_complexity', {}).get(comp)
        if not isinstance(comp_dict, dict):
            continue
        acc = comp_dict.get('accuracy')
        exact = comp_dict.get('exact_code_match_rate')

        if isinstance(acc, (int, float)):
            acc = {"mean": float(acc), "std": None}
        if isinstance(exact, (int, float)):
            exact = {"mean": float(exact), "std": None}

        if isinstance(acc, dict) and isinstance(exact, dict) and 'mean' in acc and 'mean' in exact:
            # improvement in percentage points
            delta_mean = acc['mean'] - exact['mean']
            delta_std = None
            if isinstance(acc.get('std'), (int, float)) and isinstance(exact.get('std'), (int, float)):
                delta_std = math.sqrt(acc['std']**2 + exact['std']**2)
            comp_dict['improvement_pp'] = {"mean": delta_mean, "std": delta_std}

            # relative improvement (%)
            rel_mean = 0.0
            rel_std = None
            if exact['mean'] and exact['mean'] != 0:
                rel_mean = (delta_mean / exact['mean']) * 100.0
                if isinstance(acc.get('std'), (int, float)) and isinstance(exact.get('std'), (int, float)):
                    d_f_d_acc = 100.0 / exact['mean']
                    d_f_d_exc = -100.0 * (acc['mean'] / (exact['mean']**2))
                    rel_var = (d_f_d_acc**2) * (acc['std']**2) + (d_f_d_exc**2) * (exact['std']**2)
                    rel_std = math.sqrt(rel_var)
            comp_dict['improvement_relative'] = {"mean": rel_mean, "std": rel_std}

    return data

def parse_model_summary_text(content: str) -> dict:
    """
    Parse full multi-model text into { model_name: model_dict }.
    """
    model_pattern = r'(\n|^)([^\n]+)\n-+\n(.*?)(?=\n={70,}|\Z)'
    sections = re.findall(model_pattern, content, re.DOTALL)
    out: dict = {}
    for _, model_name, section_content in sections:
        out[model_name.strip()] = _parse_model_section_to_dict(section_content)
    return out

def convert_model_summary_txt_to_json(txt_path: str | Path) -> Path:
    """
    Convert the multi-model .txt file into structured JSON next to it.
    Returns output JSON path.
    """
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    with p.open('r', encoding='utf-8') as fh:
        content = fh.read()
    data = parse_model_summary_text(content)
    out_path = p.with_suffix('.json')
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert multi-model summary .txt to .json")
    parser.add_argument('input', help="Path to the model comparison summary .txt file")
    args = parser.parse_args()
    out = convert_model_summary_txt_to_json(args.input)
    print(f"Wrote JSON: {out}")

if __name__ == '__main__':
    main()
