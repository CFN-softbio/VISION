import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from pathlib import Path
import json
import argparse
from typing import List
import seaborn as sns
import pandas as pd

# Set style for better visualization
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 600

def load_human_runs(human_paths: List[str]) -> dict:
    """
    Read one‐or-more aggregated_statistics.json files coming from
    beam-line-scientist (human-run) evaluations and convert them to the
    same models_data structure that parse_model_data() produces.
    Each human run is registered under a model name that is extracted
    from the file path (bs_1, bs_2, …).  Only complex-task metrics are
    present, so the ‘simple’ entry is left empty.
    """
    human_data = {}
    for path in human_paths:
        p = Path(path)
        # Accept a directory or direct file path
        stats_file = p / 'aggregated_statistics.json' if p.is_dir() else p
        if not stats_file.exists():
            print(f"WARNING: human run file not found: {stats_file} – skipped")
            continue

        with stats_file.open() as fh:
            stats_json = json.load(fh)

        complex_stats = (
            stats_json.get('metrics_by_complexity', [{}])[0]
            .get('complex', {})
        )
        if not complex_stats:
            print(f"WARNING: no complex metrics in {stats_file} – skipped")
            continue

        # Determine a readable model name (e.g. bs_2)
        m = re.search(r'(bs_[0-9]+)', str(stats_file))
        model_name = m.group(1) if m else stats_file.parent.name

        exact_code_rate = float(complex_stats.get('exact_code_match_rate', 0))
        accuracy = float(complex_stats.get('accuracy', 0))
        avg_full_score = float(complex_stats.get('average_full_score', 0))
        avg_pv_match_rate = float(complex_stats.get('average_pv_match_rate', 0))

        human_data[model_name] = {
            'simple': {},                     # no simple data
            'complex': {
                'exact_code_match_rate': exact_code_rate,
                'accuracy': accuracy,
                'improvement_pp': accuracy - exact_code_rate,
                'improvement_relative':
                    ((accuracy - exact_code_rate) / exact_code_rate * 100)
                    if exact_code_rate else 0,
                'average_full_score': avg_full_score * 100,
                'average_pv_match_rate': avg_pv_match_rate,
            },
        }
    return human_data

def parse_model_data(file_path):
    """Parse the model comparison summary text file to extract improvement metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    models_data = {}
    
    # Find all model sections by looking for model name followed by dashes
    # Each model section ends with a long line of equals signs OR the end of file
    model_pattern = r'(\n|^)([a-zA-Z0-9.-]+)\n-+\n(.*?)(?=\n={70,}|\Z)'
    matches = re.findall(model_pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} model sections")
    
    for _, model_name, section_content in matches:
        print(f"\nProcessing model: {model_name}")
        
        # Initialize data structure for this model
        models_data[model_name] = {
            'simple': {},
            'complex': {}
        }
        
        # The section_content includes everything from after the model name until the next separator
        # This should include both the statistics summary and the metrics_by_complexity sections
        
        # Extract simple complexity metrics
        simple_match = re.search(r'simple:.*?(?=complex:|$)', section_content, re.DOTALL)
        if simple_match:
            simple_text = simple_match.group(0)
            
            # Extract exact_code_match_rate
            code_match = re.search(r'exact_code_match_rate:\s*([\d.]+)', simple_text)
            # Extract accuracy
            accuracy_match = re.search(r'accuracy:\s*([\d.]+)', simple_text)
            
            if code_match and accuracy_match:
                exact_code_rate = float(code_match.group(1))
                accuracy = float(accuracy_match.group(1))
                
                print(f"  Simple - exact_code_match_rate: {exact_code_rate}")
                print(f"  Simple - accuracy: {accuracy}")
                
                models_data[model_name]['simple'] = {
                    'exact_code_match_rate': exact_code_rate,
                    'accuracy': accuracy,
                    'improvement_pp': accuracy - exact_code_rate,  # percentage points
                    'improvement_relative': ((accuracy - exact_code_rate) / exact_code_rate * 100) if exact_code_rate > 0 else 0  # relative %
                }
                
                # Extract average_full_score
                full_score_match = re.search(r'average_full_score:\s*([\d.]+)', simple_text)
                if full_score_match:
                    avg_fs = float(full_score_match.group(1)) * 100
                    models_data[model_name]['simple']['average_full_score'] = avg_fs
                # Extract average_pv_match_rate
                pv_match = re.search(r'average_pv_match_rate:\s*([\d.]+)', simple_text)
                if pv_match:
                    pv_rate = float(pv_match.group(1))      # value already is a percentage
                    models_data[model_name]['simple']['average_pv_match_rate'] = pv_rate
                else:
                    models_data[model_name]['simple']['average_pv_match_rate'] = 0.0
            else:
                models_data[model_name]['simple']['average_pv_match_rate'] = 0.0
                print(f"  Could not find simple metrics for {model_name}")
        
        # Extract complex complexity metrics
        complex_match = re.search(r'complex:(.*?)(?=\n\S|\Z)', section_content, re.DOTALL)
        if complex_match:
            complex_text = complex_match.group(0)
            
            # Extract exact_code_match_rate
            code_match = re.search(r'exact_code_match_rate:\s*([\d.]+)', complex_text)
            # Extract accuracy
            accuracy_match = re.search(r'accuracy:\s*([\d.]+)', complex_text)
            
            if code_match and accuracy_match:
                exact_code_rate = float(code_match.group(1))
                accuracy = float(accuracy_match.group(1))
                
                print(f"  Complex - exact_code_match_rate: {exact_code_rate}")
                print(f"  Complex - accuracy: {accuracy}")
                
                models_data[model_name]['complex'] = {
                    'exact_code_match_rate': exact_code_rate,
                    'accuracy': accuracy,
                    'improvement_pp': accuracy - exact_code_rate,  # percentage points
                    'improvement_relative': ((accuracy - exact_code_rate) / exact_code_rate * 100) if exact_code_rate > 0 else 0  # relative %
                }
                
                # Extract average_full_score
                full_score_match = re.search(r'average_full_score:\s*([\d.]+)', complex_text)
                if full_score_match:
                    avg_fs = float(full_score_match.group(1)) * 100
                    models_data[model_name]['complex']['average_full_score'] = avg_fs
                # Extract average_pv_match_rate
                pv_match = re.search(r'average_pv_match_rate:\s*([\d.]+)', complex_text)
                if pv_match:
                    pv_rate = float(pv_match.group(1))
                    models_data[model_name]['complex']['average_pv_match_rate'] = pv_rate
                else:
                    models_data[model_name]['complex']['average_pv_match_rate'] = 0.0
            else:
                models_data[model_name]['complex']['average_pv_match_rate'] = 0.0
                print(f"  Could not find complex metrics for {model_name}")
        else:
            print(f"  No complex section found for {model_name}")
    
    # Remove models that don't have either simple or complex data
    models_to_remove = []
    for model_name, data in models_data.items():
        if not data['simple'] and not data['complex']:
            models_to_remove.append(model_name)
            print(f"Removing {model_name} - missing simple and complex data")
    
    for model_name in models_to_remove:
        del models_data[model_name]
    
    print(f"\nTotal models with complete data: {len(models_data)}")
    
    return models_data

def plot_metrics_seaborn(models_data, output_dir, complexity='simple'):
    """
    For every model draw THREE bars (grouped side-by-side):
        • Accuracy  =  Exact-match (red)  +  Accuracy-improvement (green)  -> stacked
        • Average PV-match rate  (yellow)
        • Average Full Score     (blue)

    X-axis is ordered by Average Full Score (descending).
    """
    # ---------- collect & sort ------------------------------------------------
    records = []
    for model, d in models_data.items():
        if not d[complexity]:
            continue
        rec = d[complexity]
        records.append(
            (
                model,
                rec.get('exact_code_match_rate', 0.0),
                rec.get('improvement_pp', 0.0),
                rec.get('average_pv_match_rate', 0.0),
                rec.get('average_full_score', 0.0),
            )
        )
    if not records:
        return None

    # sort by average full score (index 4)
    records.sort(key=lambda r: r[4], reverse=True)
    models, exact, improve, pv_rate, full_score = zip(*records)

    # ---------- plotting ------------------------------------------------------
    n = len(models)
    bar_w = 0.22
    idx   = np.arange(n)
    fig_w = max(24, n * 3)          # wider canvas – 3 in per model, ≥ 24 in
    fig, ax = plt.subplots(figsize=(fig_w, 9))

    # Full-score (left)
    ax.bar(idx - bar_w, full_score, width=bar_w,
           color='#3498db', label='Average Full Score (%)')

    # Stacked Accuracy = exact + improvement (center)
    ax.bar(idx, exact,   width=bar_w, color='#e74c3c',
           label='Exact Match (%)')
    ax.bar(idx, improve, width=bar_w, bottom=exact, color='#2ecc71',
           label='Accuracy Improvement (pp)')

    # PV-match (right)
    ax.bar(idx + bar_w, pv_rate, width=bar_w,
           color='#f1c40f', label='Average PV-Match (%)')

    # ---------- cosmetics -----------------------------------------------------
    ax.set_xticks(idx)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)', weight='bold')
    ax.set_xlabel('Models',        weight='bold')
    ax.set_title(f'Model performance – {complexity.capitalize()} tasks',
                 weight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Show a single legend entry per colour
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right', title='Metric')

    # Annotate bars (skip very small bars)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + h + 1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    basefile = output_dir / f'model_metrics_{complexity}'
    for ext in ('png', 'svg'):
        fig.savefig(basefile.with_suffix(f'.{ext}'), dpi=600, bbox_inches='tight')
    return fig

def find_project_root():
    """Find the project root directory by looking for .git folder"""
    current_dir = Path.cwd()
    
    # Try to find .git directory by walking up the directory tree
    while current_dir != current_dir.parent:
        if (current_dir / '.git').exists():
            return current_dir
        current_dir = current_dir.parent
    
    # If we can't find it, just use the current directory
    return Path.cwd()

def create_summary_table(models_data, output_dir, complexity='simple'):
    """Create a summary table showing the metrics."""
    # Sort by accuracy
    sorted_models = dict(sorted(models_data.items(), 
                               key=lambda x: x[1][complexity]['accuracy'] if x[1][complexity] else 0, 
                               reverse=True))
    
    # Create figure for table with more space
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Model', 'Code\nMatch %', 'Accuracy %', 'Improvement\n(pp)', 'Relative\nImprovement %', 'Avg\nFS', 'Avg PV-Match %']
    rows = []
    
    for model, data in sorted_models.items():
        rows.append([
            model,
            f"{data[complexity]['exact_code_match_rate']:.1f}",
            f"{data[complexity]['accuracy']:.1f}",
            f"+{data[complexity]['improvement_pp']:.1f}",
            f"+{data[complexity]['improvement_relative']:.1f}%",
            f"{data[complexity].get('average_full_score', 0):.1f}",
            f"{data[complexity].get('average_pv_match_rate', 0):.1f}"
        ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.21, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title(f'Model Performance Summary ({complexity.capitalize()} Tasks): Impact of PV Matching\n' +
             '(pp = percentage points)', 
             fontweight='bold', fontsize=16, pad=20)
    
    # Save figure
    basefile = output_dir / f'model_improvement_table_{complexity}'
    for ext in ('png', 'svg'):
        plt.savefig(basefile.with_suffix(f'.{ext}'), dpi=600, bbox_inches='tight')
    
    return fig

def main():
    # argparse for optional human runs
    parser = argparse.ArgumentParser(description="Plot model / human performance.")
    parser.add_argument('--input', '-i', type=str,
                        help="Path to model_comparison_summary .txt file "
                             "(defaults to the project file).")
    parser.add_argument('--human', action='append', default=[],
                        help="Path to human aggregated_statistics.json or its parent folder "
                             "(can be given multiple times).")
    args = parser.parse_args()
    project_root = find_project_root()

    default_input = project_root / \
        'tests/results/op_cog/model_comparison_summary.txt'

    input_file = Path(args.input) if args.input else default_input

    output_dir = project_root / 'tests/results/op_cog'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for input file at: {input_file}")

    if not input_file.exists():
        print(f"ERROR: Input file not found at {input_file}")
        return
    
    # Parse the data
    models_data = parse_model_data(input_file)
    human_models = load_human_runs(args.human)
    models_data.update(human_models)
    
    if not models_data:
        print("ERROR: No model data was extracted from the file!")
        return
    
    print("\nExtracted model data:")
    print("-" * 50)
    
    # Create plots for both simple and complex tasks
    for complexity in ['simple', 'complex']:
        print(f"\n{complexity.upper()} TASKS:")
        print("-" * 30)
        
        # Check if all models have data for this complexity
        valid_models = {k: v for k, v in models_data.items() if v[complexity]}
        
        if not valid_models:
            print(f"No data available for {complexity} tasks")
            continue
            
        for model, data in valid_models.items():
            print(f"{model}:")
            print(f"  Exact Code Match Rate: {data[complexity]['exact_code_match_rate']:.1f}%")
            print(f"  Accuracy (Code + PV): {data[complexity]['accuracy']:.1f}%")
            print(f"  Improvement: +{data[complexity]['improvement_pp']:.1f}pp (+{data[complexity]['improvement_relative']:.1f}% relative)")
            if 'average_full_score' in data[complexity]:
                print(f"  Average Full Score: {data[complexity]['average_full_score']:.2f}%")
            if 'average_pv_match_rate' in data[complexity]:
                print(f"  Average PV-Match Rate: {data[complexity]['average_pv_match_rate']:.2f}%")
        
        # Create seaborn barplot for this complexity level
        if not valid_models:
            print(f"No valid models for {complexity} – skipping plots.")
            continue

        metrics_fig = plot_metrics_seaborn(valid_models, output_dir, complexity)
        table_fig   = create_summary_table(valid_models,  output_dir, complexity)
        
        print(f"\nPlots saved to {output_dir} for {complexity} tasks:")
        for ext in ('png', 'svg'):
            print(f"  - model_metrics_{complexity}.{ext}")
            print(f"  - model_improvement_table_{complexity}.{ext}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
