import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from pathlib import Path
import json
import argparse
from typing import List

# Set style for better visualization
plt.style.use('ggplot')
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
            else:
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
            else:
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

def plot_improvement_stacked_bar(models_data, output_dir, complexity='simple'):
    """
    Create stacked bar chart showing improvement from code matching to full accuracy,
    with an additional side-by-side bar for Average Full Score (FS).
    """
    # Sort models by Average Full Score (descending)
    sorted_models = dict(sorted(models_data.items(),
                               key=lambda x: x[1][complexity].get('average_full_score', 0)
                               if x[1][complexity] else 0,
                               reverse=True))

    # Prepare data for plotting
    model_names = list(sorted_models.keys())
    exact_code_rates = [data[complexity].get('exact_code_match_rate', 0) for data in sorted_models.values()]
    improvements = [data[complexity].get('improvement_pp', 0) for data in sorted_models.values()]
    avg_full_scores = [data[complexity].get('average_full_score', 0) for data in sorted_models.values()]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up positions for bars
    # wider gaps between model groups and between the two bars
    spacing_factor = 1.4                      # distance between model groups
    x = np.arange(len(model_names)) * spacing_factor

    width = 0.30                              # slightly thinner bars
    inner_gap = 0.4                          # was 0.05 – gives visibly more space
    fs_x      = x - (width/2 + inner_gap/2)   # left  – full-score bar
    stacked_x = x + (width/2 + inner_gap/2)   # right – red/green stack

    # New full-score bar (drawn first, on the left)
    bars_fs = ax.bar(
        fs_x,
        avg_full_scores,
        width,
        label='Average Full Score (FS)',
        color='#3498db',
        alpha=0.8,
    )
    # Create stacked bars (drawn second, on the right)
    bars1 = ax.bar(stacked_x, exact_code_rates, width, label='Exact Code Match Rate',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(stacked_x, improvements, width, bottom=exact_code_rates,
                   label='Accuracy (PV matching, timing, and/or temperature) (pp)',
                   color='#2ecc71', alpha=0.8)

    # Legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Model Performance ({complexity.capitalize()} Tasks): From Code Matching to Full Accuracy\n' + 
                 '(Showing percentage point improvement with simulator-based PV matching)',
                 fontweight='bold', fontsize=16, pad=20)
    # X-Ticks (centred between the two bars)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add extra head-room for the SIMPLE case (labels won’t clash at the top)
    ylim_top = 110 if complexity == 'simple' else 100
    ax.set_ylim(0, ylim_top)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Horizontal offset for the two top labels
    label_shift = inner_gap / 2      # horizontal offset for the two top labels

    # Add value labels on bars (including FS bar)
    for i, (bar1, bar2, bar_fs) in enumerate(zip(bars1, bars2, bars_fs)):
        # Label for exact code match rate
        height1 = bar1.get_height()
        if height1 > 5:  # Only show label if bar is tall enough
            ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                   f'{height1:.1f}%',
                   ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold')
        
        # Label for improvement
        height2 = bar2.get_height()
        if height2 > 5:  # Only show label if bar is tall enough
            ax.text(bar2.get_x() + bar2.get_width()/2., 
                   height1 + height2/2,
                   f'+{height2:.1f}pp',
                   ha='center', va='center', fontsize=10,
                   color='white', fontweight='bold')
        
        # Total accuracy label on top
        total = height1 + height2
        ax.text(bar2.get_x() + bar2.get_width()/2. + label_shift, 
               min(total + 1, ylim_top - 2),
               f'{total:.1f}%',
               ha='center', va='bottom', fontsize=11,
               fontweight='bold')

        # Full-score label on top of its bar
        fs_height = bar_fs.get_height()
        if fs_height > 5:
            ax.text(
                bar_fs.get_x() + bar_fs.get_width()/2. - label_shift,
                min(fs_height + 1, ylim_top - 2),
                f'{fs_height:.1f}%',
                ha='center', va='bottom', fontsize=10,
                color='blue', fontweight='bold',
            )
    
    # Add horizontal line at 50% for reference
    ax.axhline(y=50, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.02, 51, '50%', transform=ax.transData, fontsize=10, alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'model_improvement_stacked_bar_{complexity}.png', dpi=600, bbox_inches='tight')
    
    return fig

def plot_improvement_comparison(models_data, output_dir, complexity='simple'):
    """Create a grouped bar chart comparing old vs new metrics side by side."""
    # Sort models by improvement amount (descending)
    sorted_models = dict(sorted(models_data.items(), 
                               key=lambda x: x[1][complexity]['improvement_pp'] if x[1][complexity] else 0, 
                               reverse=True))
    
    # Prepare data for plotting
    model_names = list(sorted_models.keys())
    exact_code_rates = [data[complexity].get('exact_code_match_rate', 0) for data in sorted_models.values()]
    accuracies = [data[complexity].get('accuracy', 0) for data in sorted_models.values()]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up positions for bars
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, exact_code_rates, width, 
                   label='Exact Code Match Rate (Old Metric)',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, accuracies, width,
                   label='Accuracy (Code + PV Match)',
                   color='#2ecc71', alpha=0.8)
    
    # Add improvement arrows
    for i in range(len(model_names)):
        if accuracies[i] > exact_code_rates[i]:
            ax.annotate('', xy=(x[i] + width/2, accuracies[i]),
                       xytext=(x[i] - width/2, exact_code_rates[i]),
                       arrowprops=dict(arrowstyle='->', color='black', 
                                     alpha=0.5, lw=1.5))
            # Add improvement percentage
            improvement = accuracies[i] - exact_code_rates[i]
            mid_height = (accuracies[i] + exact_code_rates[i]) / 2
            ax.text(x[i], mid_height, f'+{improvement:.1f}pp',
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='yellow', alpha=0.7))
    
    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=14)
    ax.set_title(f'Comparison ({complexity.capitalize()} Tasks): Code-Only Matching vs Full Accuracy (Code + PV)\n' +
                 '(Showing percentage point differences)',
                fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 100)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'model_improvement_comparison_{complexity}.png', dpi=600, bbox_inches='tight')
    
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
    headers = ['Model', 'Code\nMatch %', 'Accuracy %', 'Improvement\n(pp)', 'Relative\nImprovement %', 'Avg\nFS']
    rows = []
    
    for model, data in sorted_models.items():
        rows.append([
            model,
            f"{data[complexity]['exact_code_match_rate']:.1f}",
            f"{data[complexity]['accuracy']:.1f}",
            f"+{data[complexity]['improvement_pp']:.1f}",
            f"+{data[complexity]['improvement_relative']:.1f}%",
            f"{data[complexity].get('average_full_score', 0):.1f}"
        ])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
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
    plt.savefig(output_dir / f'model_improvement_table_{complexity}.png', dpi=600, bbox_inches='tight')
    
    return fig

def main():
    # argparse for optional human runs
    parser = argparse.ArgumentParser(description="Plot model / human performance.")
    parser.add_argument('--human', action='append',
                        help="Path to human aggregated_statistics.json "
                             "or its parent folder. Can be given multiple times.",
                        default=[])
    args = parser.parse_args()
    # Find project root
    project_root = find_project_root()
    
    # Set paths
    # First try the specific file mentioned in the user's output
    input_file = project_root / 'tests/results/op_cog/model_comparison_summary_new_dataset_old_prompt_16_7_with_new_models_manually_added.txt'
    
    # If that doesn't exist, try the generic name
    if not input_file.exists():
        raise FileNotFoundError(input_file)
        input_file = project_root / 'tests/results/op_cog/model_comparison_summary.txt'
    
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
        
        # Create plots for this complexity level
        if not valid_models:
            print(f"No valid models for {complexity} – skipping plots.")
            continue
        stacked_fig    = plot_improvement_stacked_bar(valid_models, output_dir, complexity)
        comparison_fig = plot_improvement_comparison(valid_models, output_dir, complexity)
        table_fig      = create_summary_table(valid_models,  output_dir, complexity)
        
        print(f"\nPlots saved to {output_dir} for {complexity} tasks:")
        print(f"  - model_improvement_stacked_bar_{complexity}.png")
        print(f"  - model_improvement_comparison_{complexity}.png")
        print(f"  - model_improvement_table_{complexity}.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
