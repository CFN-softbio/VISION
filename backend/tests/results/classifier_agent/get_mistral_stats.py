import os
import json
from datetime import datetime
import numpy as np

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_statistics(data):
    stats = {}
    for key in data:
        values = np.array(data[key])
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return stats

def format_model_stats(model_name, stats):
    # Create the model header
    header = f"\n{model_name}\n" + "-" * len(model_name) + "\n"
    
    # Create the statistics section
    stats_header = "Statistics Summary for classifier agent (mean ± std):\n"
    separator = "=" * 50 + "\n"
    
    # Format each statistic
    stats_lines = []
    for key in ['accuracy', 'total_entries', 'correct_matches', 'average_execution_time', 'f1_score']:
        mean = stats[key]['mean']
        std = stats[key]['std']
        
        # Format with appropriate precision
        if key in ['accuracy', 'total_entries', 'correct_matches']:
            formatted_line = f"{key}: {mean:.4f} ± {std:.4f}"
        else:
            formatted_line = f"{key}: {mean:.4f} ± {std:.4f}"
        stats_lines.append(formatted_line)
    
    stats_body = "\n".join(stats_lines)
    
    # Combine all parts
    return (
        header +
        stats_header +
        separator +
        stats_body + "\n" +
        separator + "\n" +
        "=" * 80 + "\n"
    )

def generate_comparison_summary(base_path):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Start building the summary
    summary = [
        f"CLASSIFIER Agent Model Comparison Summary - {timestamp}",
        "=" * 80,
        "\n"
    ]
    
    # Get all model directories
    model_dirs = ['mistral']
    
    # Process each model
    for model_dir in sorted(model_dirs):
        model_path = os.path.join(base_path, model_dir)
        # Get the first (and presumably only) subdirectory
        subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        if subdirs:  # If there are subdirectories
            subdir = subdirs[0]  # Take the first subdirectory
            stats_file = os.path.join(model_path, subdir, 'aggregated_statistics.json')
            
            if os.path.exists(stats_file):
                # Read and process statistics
                data = read_json_file(stats_file)
                stats = calculate_statistics(data)
                
                # Format and add to summary
                summary.append(format_model_stats(model_dir, stats))
    
    # Write the summary file
    output_path = os.path.join(base_path, 'model_comparison_summary_mistral.txt')
    with open(output_path, 'w') as f:
        f.write(''.join(summary))
    
    print(f"Summary written to: {output_path}")

# Usage
base_path = 'list_output/'
generate_comparison_summary(base_path)