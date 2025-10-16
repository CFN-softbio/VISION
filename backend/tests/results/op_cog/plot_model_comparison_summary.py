import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# Set style for better visualization
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def parse_model_data(file_path):
    """Parse the model comparison summary text file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Dictionary to store model data
    models_data = {}
    
    # Split content by model sections
    sections = re.split(r'\n={10,}\n+', content)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract model name
        model_match = re.search(r'([a-zA-Z0-9.-]+)\n-+', section)
        if not model_match:
            continue
            
        model_name = model_match.group(1).strip()
        print(f"Found model section: {model_name}")
        
        # Extract metrics - need to handle the indentation and formatting in the file
        # Look for metrics in the complex section which is nested under metrics_by_complexity
        metrics_section = re.search(r'metrics_by_complexity:.*?complex:.*?(?=\n\s*accuracy|\Z)', section, re.DOTALL)
        if not metrics_section:
            print(f"  No metrics section found for {model_name}")
            continue
            
        metrics_text = metrics_section.group(0)
        
        # Find metrics with proper indentation patterns
        pv_match = re.search(r'pv_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', metrics_text)
        timing_match = re.search(r'timing_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', metrics_text)
        temp_match = re.search(r'temp_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', metrics_text)
        
        # Execution time might be in a different section
        exec_time = re.search(r'average_execution_time:\s*([\d.]+)\s*±\s*([\d.]+)', section)
        
        # Debug prints
        if pv_match:
            print(f"  Found pv_match_rate: {pv_match.group(1)} ± {pv_match.group(2)}")
        if timing_match:
            print(f"  Found timing_match_rate: {timing_match.group(1)} ± {timing_match.group(2)}")
        if temp_match:
            print(f"  Found temp_match_rate: {temp_match.group(1)} ± {temp_match.group(2)}")
        if exec_time:
            print(f"  Found execution_time: {exec_time.group(1)} ± {exec_time.group(2)}")
        
        # Only add if we have the required metrics for plotting
        if pv_match and timing_match and temp_match:
            models_data[model_name] = {
                'pv_match_rate': (float(pv_match.group(1)), float(pv_match.group(2))),
                'timing_match_rate': (float(timing_match.group(1)), float(timing_match.group(2))),
                'temp_match_rate': (float(temp_match.group(1)), float(temp_match.group(2))),
                'execution_time': (float(exec_time.group(1)), float(exec_time.group(2))) if exec_time else (0, 0)
            }
    
    return models_data

def plot_match_rates(models_data, output_dir):
    """Create bar charts for match rates."""
    # Sort models by average of pv and timing match rates
    def sort_key(item):
        model_name, data = item
        return (data['pv_match_rate'][0] + data['timing_match_rate'][0]) / 2
    
    sorted_models = dict(sorted(models_data.items(), key=sort_key, reverse=True))
    
    # Prepare data for plotting
    model_names = list(sorted_models.keys())
    pv_rates = [data['pv_match_rate'][0] for data in sorted_models.values()]
    pv_stds = [data['pv_match_rate'][1] for data in sorted_models.values()]
    timing_rates = [data['timing_match_rate'][0] for data in sorted_models.values()]
    timing_stds = [data['timing_match_rate'][1] for data in sorted_models.values()]
    temp_rates = [data['temp_match_rate'][0] for data in sorted_models.values()]
    temp_stds = [data['temp_match_rate'][1] for data in sorted_models.values()]
    
    # Set up positions for bars
    x = np.arange(len(model_names))
    width = 0.25
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot bars
    rects1 = ax.bar(x - width, pv_rates, width, yerr=pv_stds, label='PV Match Rate', 
                   color='#3498db', alpha=0.8, capsize=5)
    rects2 = ax.bar(x, timing_rates, width, yerr=timing_stds, label='Timing Match Rate', 
                   color='#2ecc71', alpha=0.8, capsize=5)
    rects3 = ax.bar(x + width, temp_rates, width, yerr=temp_stds, label='Temperature Match Rate', 
                   color='#e74c3c', alpha=0.8, capsize=5)
    
    # Add labels, title and legend
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Match Rate (%)', fontweight='bold', fontsize=14)
    ax.set_title('Model Performance Comparison - Match Rates', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower left', fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 110)  # Give a little space above 100%
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'model_match_rates.png', dpi=300, bbox_inches='tight')
    
    return fig

def plot_execution_times(models_data, output_dir):
    """Create bar chart for execution times."""
    # Sort models by execution time
    sorted_models = dict(sorted(models_data.items(), key=lambda x: x[1]['execution_time'][0]))
    
    # Prepare data for plotting
    model_names = list(sorted_models.keys())
    exec_times = [data['execution_time'][0] for data in sorted_models.values()]
    exec_stds = [data['execution_time'][1] for data in sorted_models.values()]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot bars
    bars = ax.bar(model_names, exec_times, yerr=exec_stds, 
                 color='#9b59b6', alpha=0.8, capsize=5)
    
    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=14)
    ax.set_title('Model Performance Comparison - Average Execution Time', 
                fontweight='bold', fontsize=16)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'model_execution_times.png', dpi=300, bbox_inches='tight')
    
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

def plot_combined_metrics(models_data, output_dir):
    """Create a combined plot with all metrics."""
    # Sort models by average of pv and timing match rates
    def sort_key(item):
        model_name, data = item
        return (data['pv_match_rate'][0] + data['timing_match_rate'][0]) / 2
    
    sorted_models = dict(sorted(models_data.items(), key=sort_key, reverse=True))
    
    # Prepare data for plotting
    model_names = list(sorted_models.keys())
    metrics = ['pv_match_rate', 'timing_match_rate', 'temp_match_rate']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), height_ratios=[3, 1])
    
    # Plot match rates on top subplot
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [data[metric][0] for data in sorted_models.values()]
        errors = [data[metric][1] for data in sorted_models.values()]
        
        rects = ax1.bar(x + (i-1)*width, values, width, yerr=errors, 
                       label=metric.replace('_', ' ').title(), 
                       color=colors[i], alpha=0.8, capsize=5)
        
        # Add value labels
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Configure top subplot
    ax1.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Match Rate (%)', fontweight='bold', fontsize=14)
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot execution times on bottom subplot
    exec_times = [data['execution_time'][0] for data in sorted_models.values()]
    exec_errors = [data['execution_time'][1] for data in sorted_models.values()]
    
    bars = ax2.bar(model_names, exec_times, yerr=exec_errors, 
                  color='#9b59b6', alpha=0.8, capsize=5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Configure bottom subplot
    ax2.set_xlabel('Models', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Execution Time (s)', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'model_combined_metrics.png', dpi=300, bbox_inches='tight')
    
    return fig

def parse_model_data_manual(file_path):
    """Alternative parsing method that uses a more direct approach."""
    models_data = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_model = None
    
    for i, line in enumerate(lines):
        # Look for model name headers
        if re.match(r'^[a-zA-Z0-9.-]+$', line.strip()) and i+1 < len(lines) and re.match(r'^-+$', lines[i+1].strip()):
            current_model = line.strip()
            print(f"Manual parsing - found model: {current_model}")
            
        # Look for specific metrics
        if current_model:
            # Match rate metrics
            if 'pv_match_rate:' in line:
                match = re.search(r'pv_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match and current_model not in models_data:
                    models_data[current_model] = {}
                    models_data[current_model]['pv_match_rate'] = (float(match.group(1)), float(match.group(2)))
                    print(f"  Manual - found pv_match_rate: {match.group(1)} ± {match.group(2)}")
            
            if 'timing_match_rate:' in line:
                match = re.search(r'timing_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match and current_model in models_data:
                    models_data[current_model]['timing_match_rate'] = (float(match.group(1)), float(match.group(2)))
                    print(f"  Manual - found timing_match_rate: {match.group(1)} ± {match.group(2)}")
            
            if 'temp_match_rate:' in line:
                match = re.search(r'temp_match_rate:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match and current_model in models_data:
                    models_data[current_model]['temp_match_rate'] = (float(match.group(1)), float(match.group(2)))
                    print(f"  Manual - found temp_match_rate: {match.group(1)} ± {match.group(2)}")
            
            # Execution time might be in a different section
            if 'average_execution_time:' in line:
                match = re.search(r'average_execution_time:\s*([\d.]+)\s*±\s*([\d.]+)', line)
                if match and current_model in models_data:
                    models_data[current_model]['execution_time'] = (float(match.group(1)), float(match.group(2)))
                    print(f"  Manual - found execution_time: {match.group(1)} ± {match.group(2)}")
    
    # Filter out models that don't have all required metrics
    complete_models = {}
    for model, data in models_data.items():
        if all(key in data for key in ['pv_match_rate', 'timing_match_rate', 'temp_match_rate']):
            if 'execution_time' not in data:
                data['execution_time'] = (0, 0)
            complete_models[model] = data
    
    return complete_models

def main():
    # Find project root
    project_root = find_project_root()
    
    # Set paths
    input_file = project_root / 'tests/results/op_cog/model_comparison_summary.txt'
    output_dir = project_root / 'tests/results/op_cog'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for input file at: {input_file}")
    
    # Try regular parsing first
    models_data = parse_model_data(input_file)
    
    # If that fails, try manual parsing
    if not models_data:
        print("Regular parsing failed, trying manual parsing...")
        models_data = parse_model_data_manual(input_file)
    
    # Debug: print extracted data
    print("Extracted model data:")
    for model, data in models_data.items():
        print(f"{model}:")
        for metric, values in data.items():
            print(f"  {metric}: {values}")
    
    if not models_data:
        print("ERROR: No model data was extracted from the file!")
        return
    
    # Create plots
    match_rates_fig = plot_match_rates(models_data, output_dir)
    exec_times_fig = plot_execution_times(models_data, output_dir)
    combined_fig = plot_combined_metrics(models_data, output_dir)
    
    print(f"Plots saved to {output_dir}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
