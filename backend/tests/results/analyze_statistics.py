import json
import numpy as np
from typing import Dict, Any, Union
import argparse
from pathlib import Path

def calculate_stats(values: list) -> Dict[str, float]:
    """Calculate mean and standard deviation for a list of numeric values."""
    if not values:
        return {'mean': 0, 'std': 0}
    
    # Convert string numbers to float if needed
    numeric_values = [float(v) if isinstance(v, str) and v.replace('.', '').isdigit() else v 
                     for v in values if isinstance(v, (int, float)) or 
                     (isinstance(v, str) and v.replace('.', '').isdigit())]
    
    if not numeric_values:
        return {'mean': 0, 'std': 0}
        
    return {
        'mean': np.mean(numeric_values),
        'std': np.std(numeric_values, ddof=1) if len(numeric_values) > 1 else 0  # ddof=1 for sample std
    }

def process_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process nested dictionaries and calculate statistics."""
    def process_value(v):
        if isinstance(v, dict):
            return process_nested_dict(v)
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                return {k: process_value([item[k] for item in v]) 
                       for k in v[0].keys()}
            else:
                return calculate_stats(v)
        return v

    return {k: process_value(v) for k, v in d.items()}

def format_stats(stats: Dict[str, Any], indent: int = 0) -> str:
    """Format statistics into a readable string with proper indentation."""
    lines = []
    indent_str = "  " * indent
    
    for key, value in stats.items():
        if isinstance(value, dict):
            if 'mean' in value and 'std' in value:
                # Format as mean ± std
                lines.append(f"{indent_str}{key}: {value['mean']:.4f} ± {value['std']:.4f}")
            else:
                # Nested dictionary
                lines.append(f"{indent_str}{key}:")
                lines.extend(format_stats(value, indent + 1).split('\n'))
        else:
            lines.append(f"{indent_str}{key}: {value}")
            
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Analyze aggregated statistics from test runs.')
    parser.add_argument('stats_file', type=str, help='Path to the aggregated_statistics.json file')
    parser.add_argument('--agent_type', type=str, required=True, 
                      choices=['op', 'ana', 'classifier'],
                      help='Type of agent to analyze')
    args = parser.parse_args()
    
    stats_path = Path(args.stats_file)
    if not stats_path.exists():
        print(f"Error: File {stats_path} does not exist")
        return
        
    with open(stats_path, 'r') as f:
        data = json.load(f)
        
    # Process the statistics
    stats = process_nested_dict(data)
    
    # Print formatted results
    print(f"\nStatistics Summary for {args.agent_type} agent (mean ± std):")
    print("=" * 50)
    print(format_stats(stats))
    print("=" * 50)
    
    # Also save to a file next to the input file
    output_path = stats_path.parent / 'statistics_summary.txt'
    with open(output_path, 'w') as f:
        f.write(f"Statistics Summary for {args.agent_type} agent (mean ± std):\n")
        f.write("=" * 50 + "\n")
        f.write(format_stats(stats))
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"\nSummary saved to: {output_path}")

if __name__ == '__main__':
    main()
