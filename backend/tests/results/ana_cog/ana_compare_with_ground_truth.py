import csv
import argparse
import os
import json

from codebleu import calc_codebleu

from tests.utils import print_comparison

def compare_csv_results(csv_path):
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_id, row in enumerate(reader):
            expected_codes = json.loads(row['expected_codes'])  # Parse as JSON array
            levenshtein_distances = json.loads(row['levenshtein_distances'])
            generated_code = row['generated_code']

            # Retrieve Levenshtein distances and CodeBLEU scores from CSV
            normalized_levenshtein_distances = json.loads(row['normalized_levenshtein_distances'])
            
            comparisons = [
                {
                    'expected_code': expected_code,
                    'generated_code': generated_code,
                    'levenshtein_distance': levenshtein_distances[i],
                    'normalized_levenshtein_distance': normalized_levenshtein_distances[i],
                }
                for i, expected_code in enumerate(expected_codes)
            ]

            # Sort by normalized Levenshtein distance
            comparisons.sort(key=lambda x: x['normalized_levenshtein_distance'])

            print(f"\n{row_id}: {row['command']}")
            print(f"# ground truth entries: {len(expected_codes)}")
            print("=" * 50)
            print("=" * 50)
            print()
            for comparison in comparisons:
                print_comparison(comparison['expected_code'], generated_code, 
                               normalized_levenshtein=comparison['normalized_levenshtein_distance'],
                                 should_calc_codebleu=False)

            print("=" * 50)
            print("=" * 50)
            print("\n\n")

def find_most_recent_csv(directory):
    """Find most recent CSV file in the nested directory structure."""
    def get_most_recent(path, filter_func):
        items = [os.path.join(path, d) for d in os.listdir(path) if filter_func(os.path.join(path, d))]
        return max(items, key=os.path.getmtime) if items else None
    
    model_dir = get_most_recent(directory, os.path.isdir)
    if not model_dir:
        raise FileNotFoundError("No model directories found")
        
    timestamp_dir = get_most_recent(model_dir, os.path.isdir)
    if not timestamp_dir:
        raise FileNotFoundError(f"No timestamp directories in {model_dir}")
    
    # First try to find CSV directly in timestamp directory
    csv_file = get_most_recent(timestamp_dir, lambda x: x.endswith('.csv'))
    if csv_file:
        print(f"Using results from: {csv_file}")
        return csv_file
        
    # If no CSV found, look in run subdirectories
    run_dir = get_most_recent(timestamp_dir, os.path.isdir)
    if not run_dir:
        raise FileNotFoundError(f"No CSV files or run directories in {timestamp_dir}")
        
    csv_file = get_most_recent(run_dir, lambda x: x.endswith('.csv'))
    if not csv_file:
        raise FileNotFoundError(f"No CSV files in {run_dir}")
        
    print(f"Using results from: {csv_file}")
    return csv_file


def main():
    parser = argparse.ArgumentParser(description='Compare expected and generated codes from a CSV file.')
    parser.add_argument('csv_path', type=str, nargs='?', help='Path to the CSV file containing the results')
    args = parser.parse_args()
    if args.csv_path:
        csv_path = args.csv_path
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Find the most recently modified directory under results/op_cog
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            raise FileNotFoundError("No subdirectories found in the results/ana_cog directory.")
        most_recent_subdir = max(subdirs, key=os.path.getmtime)
        csv_path = find_most_recent_csv(most_recent_subdir)

    print(f"Comparing results from {csv_path}")

    compare_csv_results(csv_path)


if __name__ == '__main__':
    main()
