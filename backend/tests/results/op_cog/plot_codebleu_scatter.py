import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from pathlib import Path
import argparse   # NEW

# Optional: for label collision avoidance
try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

# Set style for better visualization
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def parse_model_data(file_path):
    """Parse the model comparison summary text file to extract CodeBLEU, PV match rate, and full score."""
    with open(file_path, 'r') as f:
        content = f.read()

    models_data = {}

    # Find all model sections
    model_pattern = r'(\n|^)([^\n]+)\n-+\n(.*?)(?=\n={70,}|\Z)'
    matches = re.findall(model_pattern, content, re.DOTALL | re.MULTILINE)
    for _, model_name, section_content in matches:
        model_name = model_name.strip()
        if model_name.lower() == 'debug':
            continue
        models_data[model_name] = {}
        for complexity in ['simple', 'complex']:
            if complexity == 'simple':
                block_match = re.search(r'simple:.*?(?=complex:|$)', section_content, re.DOTALL | re.IGNORECASE)
            else:
                block_match = re.search(r'complex:.*?(?=\Z)', section_content, re.DOTALL | re.IGNORECASE)
            if not block_match:
                continue
            block = block_match.group(0)

            # CodeBLEU extraction (try preferred, fallback to alternates)
            codebleu = None
            for cb_pat in [
                r'comb[_\s]?7[_\s]?codebleu:\s*([\d.]+)',   # primary – matches “comb_7_CodeBLEU:”
                r'codebleu.*?comb[_\s]?7.*?([\d.]+)',       # fallback, keeps old behaviour
            ]:
                cb_match = re.search(cb_pat, block, re.IGNORECASE | re.DOTALL)
                if cb_match:
                    codebleu = float(cb_match.group(1))
                    break
            if codebleu is not None and codebleu <= 1.0:
                codebleu *= 100

            # PV match rate
            pv_match = re.search(r'average_pv_match_rate:\s*([\d.]+)', block, re.IGNORECASE)
            if not pv_match:
                pv_match = re.search(r'pv_match(?:_rate)?\s*[:=]\s*([\d.]+)', block, re.IGNORECASE)
            pv_val = float(pv_match.group(1)) if pv_match else None
            if pv_val is not None and pv_val <= 1.0:
                pv_val *= 100

            # Average full score
            fs_match = re.search(r'average_full_score:\s*([\d.]+)', block, re.DOTALL | re.IGNORECASE)
            full_score_val = float(fs_match.group(1)) if fs_match else None
            if full_score_val is not None and full_score_val <= 1.0:
                full_score_val *= 100

            # Normalized Levenshtein distance
            lev_match = re.search(
                r'average_best_normalized_levenshtein\s*[:=]\s*([\d.]+)',
                block,
                re.IGNORECASE
            )
            lev_val = float(lev_match.group(1)) if lev_match else None
            if lev_val is not None and lev_val <= 1.0:
                lev_val *= 100

            entry = {}
            if codebleu is not None:
                entry['codebleu'] = codebleu
            if pv_val is not None:
                entry['pv_match_rate'] = pv_val
            if full_score_val is not None:
                entry['full_score'] = full_score_val
            if lev_val is not None:
                entry['normalized_levenshtein'] = lev_val
            models_data[model_name][complexity] = entry if entry else None

    # Normalize structure: set empty complexities to None, remove models with no valid complexity
    models_to_remove = []
    for model, comps in list(models_data.items()):
        for c in ['simple', 'complex']:
            v = comps.get(c)
            if not v:
                comps[c] = None
        if all((comps.get(c) is None for c in ['simple', 'complex'])):
            models_to_remove.append(model)
    for model in models_to_remove:
        del models_data[model]

    return models_data

def scatter_plot(models_data, output_dir, complexity, x_key, y_key,
                 x_title, y_title, fname, *,
                 add_regression: bool = False,
                 use_connectors: bool = True,
                 add_identity: bool = False):
    """
    Scatter plot of y_key vs x_key.

    Args:
        models_data: dict of model metrics.
        output_dir: directory for saving plot output.
        complexity: 'simple' or 'complex'.
        x_key: metric key for x-axis.
        y_key: metric key for y-axis.
        x_title: x-axis label.
        y_title: y-axis label.
        fname: filename for saving.
        add_regression (bool): If True, overlay a linear regression and R².
    """
    valid_models = {
        m: d for m, d in models_data.items()
        if (complexity in d and d[complexity]
            and x_key in d[complexity]
            and y_key in d[complexity])
    }
    if not valid_models:
        print(f"No valid models for complexity={complexity}")
        return None
    x_vals = [d[complexity][x_key] for d in valid_models.values()]
    y_vals = [d[complexity][y_key] for d in valid_models.values()]
    labels = list(valid_models.keys())
    colors = ['C0' if lbl.lower().startswith('bs') else 'tab:gray' for lbl in labels]

    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(x_vals, y_vals, s=80, alpha=.8, c=colors)
    texts = []
    for xi, yi, lbl in zip(x_vals, y_vals, labels):
        txt = ax.text(xi, yi, lbl, fontsize=10, ha='right', va='bottom', zorder=3)
        # Light background so connectors don't visually cut through the text
        txt.set_bbox(dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        texts.append(txt)
    for t in texts:
        t.set_clip_on(False)


    ax.set_xlabel(f'{x_title} %', fontweight='bold')
    ax.set_ylabel(f'{y_title} %', fontweight='bold')
    ax.set_title(f'{y_title} vs {x_title} ({complexity.capitalize()} tasks)')

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.axhline(50, ls=':', color='grey', lw=1)
    ax.axvline(50, ls=':', color='grey', lw=1)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')

    if add_identity:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if x_key == 'normalized_levenshtein':
            # Axes are 0–100%, so the “inverse identity” is y = 100 - x
            xx = np.linspace(x0, x1, 200)
            yy = 100 - xx
            mask = (yy >= y0) & (yy <= y1)
            ax.plot(xx[mask], yy[mask], color='k', ls=':', lw=1, alpha=0.7, label='y = 1 - x')
        else:
            # Standard identity y = x across the overlapping visible range
            d_min = max(min(x0, x1), min(y0, y1))
            d_max = min(max(x0, x1), max(y0, y1))
            xx = np.linspace(d_min, d_max, 200)
            ax.plot(xx, xx, color='k', ls=':', lw=1, alpha=0.7, label='y = x')

    # Prevent label overlaps (uses adjustText if available; otherwise simple jitter fallback)
    if adjust_text is not None:
        adjust_text(
            texts, x=x_vals, y=y_vals, ax=ax,
            expand_text=(1.2, 1.4),
            expand_points=(2.0, 2.0),
            force_text=(0.5, 0.7),
            force_points=(0.3, 0.5),
            lim=500,
            precision=0.001
        )
    else:
        seen = {}
        for i, (xi, yi) in enumerate(zip(x_vals, y_vals)):
            key = (round(xi, 1), round(yi, 1))
            cnt = seen.get(key, 0)
            if cnt:
                offset = 0.6 * cnt  # move subsequent labels slightly to avoid exact overlaps
                texts[i].set_position((xi + offset, yi + offset))
            seen[key] = cnt + 1

    # Final small repel pass to clear any residual overlaps (in data coords)
    fig.canvas.draw()
    def _boxes_overlap(t1, t2, renderer):
        b1 = t1.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
        b2 = t2.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
        return b1.overlaps(b2)

    for _ in range(10):
        moved = False
        renderer = fig.canvas.get_renderer()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if _boxes_overlap(texts[i], texts[j], renderer):
                    xi, yi = texts[i].get_position()
                    xj, yj = texts[j].get_position()
                    dx = xi - xj
                    dy = yi - yj
                    if dx == 0 and dy == 0:
                        dx, dy = 0.5, 0.5
                    norm = np.hypot(dx, dy) or 1.0
                    step = 0.8  # move ~0.8% of axis range per iteration
                    texts[i].set_position((xi + step * dx / norm, yi + step * dy / norm))
                    texts[j].set_position((xj - step * dx / norm, yj - step * dy / norm))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()

    # Draw connectors from labels to points (manual; avoids adjustText warning)
    if use_connectors:
        for txt, xi, yi in zip(texts, x_vals, y_vals):
            x_text, y_text = txt.get_position()
            ax.annotate(
                '', xy=(xi, yi), xytext=(x_text, y_text),
                annotation_clip=False,
                arrowprops=dict(
                    arrowstyle='-',
                    color='0.5',
                    lw=0.6,
                    shrinkA=14,   # keep line from entering text bbox
                    shrinkB=6,    # gap at the point marker
                    zorder=1      # draw below text
                )
            )

    # Optional linear-fit + R²; draw line across full axis width
    if add_regression and len(x_vals) >= 2:
        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        m, b = np.polyfit(x_arr, y_arr, 1)

        x_left, x_right = ax.get_xlim()
        x_line = np.linspace(x_left, x_right, 200)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, color='blue', ls='--', label='Linear fit')

        y_pred = m * x_arr + b
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        ax.text(0.05, 0.95, f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=12, bbox=dict(fc='white', alpha=.6, ec='none'))
        ax.legend()

    # Build combined legend: keep regression if present, add category handles
    has_bs = any(lbl.lower().startswith('bs') for lbl in labels)
    has_other = any(not lbl.lower().startswith('bs') for lbl in labels)

    existing_handles, _ = ax.get_legend_handles_labels()

    category_handles = []
    if has_bs:
        category_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   label='bs_* models', markerfacecolor='C0',
                   markeredgecolor='black', markersize=9)
        )
    if has_other:
        category_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   label='Other models', markerfacecolor='tab:gray',
                   markeredgecolor='black', markersize=9)
        )

    handles = (existing_handles + category_handles) if existing_handles else category_handles
    if handles:
        ax.legend(handles=handles, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    # Also save as SVG for publication-quality vector graphics
    svg_name = Path(fname).with_suffix('.svg')
    plt.savefig(output_dir / svg_name, format='svg', bbox_inches='tight')
    return fig

def find_project_root():
    """Find the project root directory by looking for .git folder"""
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / '.git').exists():
            return current_dir
        current_dir = current_dir.parent
    return Path.cwd()

def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot CodeBLEU vs other metrics")
    parser.add_argument('--regression', '-r', action='store_true',
                        help='Add linear regression line and R² to each plot')
    parser.add_argument('--seaborn', action='store_true',
                        help='Use seaborn theme (whitegrid, paper) for publication-quality aesthetics')
    parser.add_argument('--no-connectors', dest='connectors', action='store_false',
                        help='Disable connector lines from labels to points')
    parser.add_argument('--identity', '-i', action='store_true',
                        help='Add y = x identity line to each plot')
    parser.set_defaults(connectors=True)
    args = parser.parse_args()

    if args.seaborn:
        try:
            import seaborn as sns
        except ImportError:
            print("Seaborn not installed; continuing with Matplotlib 'ggplot' style.")
        else:
            sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

    project_root = find_project_root()
    input_file = project_root / 'tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.txt'
    if not input_file.exists():
        exit(9000)
        input_file = project_root / 'tests/results/op_cog/model_comparison_summary.txt'
    output_dir = project_root / 'tests/results/op_cog'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for input file at: {input_file}")
    if not input_file.exists():
        print(f"ERROR: Input file not found at {input_file}")
        return

    models_data = parse_model_data(input_file)
    if not models_data:
        print("ERROR: No model data was extracted from the file!")
        return

    for complexity in ['simple', 'complex']:
        valid = [d for m, d in models_data.items() if complexity in d and d[complexity]]
        if not valid:
            print(f"No models with {complexity} data.")
            continue

        # CodeBLEU vs Full score
        scatter_plot(models_data, output_dir,
                     complexity,
                     x_key='codebleu',
                     y_key='full_score',
                     x_title='CodeBLEU (comb 7)',
                     y_title='Simulation (Full) score',
                     fname=f'codebleu_vs_full_{complexity}.png',
                     add_regression=args.regression,
                     use_connectors=args.connectors,
                     add_identity=args.identity)

        # CodeBLEU vs PV-matching score
        scatter_plot(models_data, output_dir,
                     complexity,
                     x_key='codebleu',
                     y_key='pv_match_rate',
                     x_title='CodeBLEU (comb 7)',
                     y_title='PV-matching score',
                     fname=f'codebleu_vs_pv_{complexity}.png',
                     add_regression=args.regression,
                     use_connectors=args.connectors,
                     add_identity=args.identity)

        # Normalized Levenshtein vs Full score
        scatter_plot(models_data, output_dir,
                     complexity,
                     x_key='normalized_levenshtein',
                     y_key='full_score',
                     x_title='Normalized Levenshtein distance',
                     y_title='Simulation (Full) score',
                     fname=f'norm_levenshtein_vs_full_{complexity}.png',
                     add_regression=args.regression,
                     use_connectors=args.connectors,
                     add_identity=args.identity)

        # Normalized Levenshtein vs PV-matching score
        scatter_plot(models_data, output_dir,
                     complexity,
                     x_key='normalized_levenshtein',
                     y_key='pv_match_rate',
                     x_title='Normalized Levenshtein distance',
                     y_title='PV-matching score',
                     fname=f'norm_levenshtein_vs_pv_{complexity}.png',
                     add_regression=args.regression,
                     use_connectors=args.connectors,
                     add_identity=args.identity)
    plt.show()

if __name__ == "__main__":
    main()
