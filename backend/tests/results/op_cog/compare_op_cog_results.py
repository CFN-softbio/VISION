import csv
import argparse
import json
import os
import pandas as pd

def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def avg_time(data, exclude_first=False):
    times = [pd.to_timedelta(row['execution_time']).total_seconds() for row in (data[1:] if exclude_first else data)]
    return sum(times) / len(times)

def compare_csv(file1, file2, dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    data1 = read_csv(file1)
    data2 = read_csv(file2)

    discrepancies = []

    for row1, row2, entry in zip(data1, data2, dataset):
        if row1['command'] != row2['command']:
            discrepancies.append(f"Commands do not match: {row1['command']} != {row2['command']}")
            continue

        if row1['exact_match'] != row2['exact_match']:
            discrepancies.append(f"Discrepancy in exact_match for command '{row1['command']}': {row1['exact_match']} != {row2['exact_match']}")
            discrepancies.append(f"Expected Code:\n{entry['expected_code']}\n\n")
            discrepancies.append(f"Produced Code (File 1):\n{row1['generated_code']}\n")
            discrepancies.append(f"Produced Code (File 2):\n{row2['generated_code']}")
            discrepancies.append("-" * 50)

    avg_time_incl_first_1 = avg_time(data1)
    avg_time_excl_first_1 = avg_time(data1, exclude_first=True)
    avg_time_incl_first_2 = avg_time(data2)
    avg_time_excl_first_2 = avg_time(data2, exclude_first=True)

    return discrepancies, avg_time_incl_first_1, avg_time_excl_first_1, avg_time_incl_first_2, avg_time_excl_first_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare exact_match between two CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file')
    parser.add_argument('file2', type=str, help='Path to the second CSV file')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset JSON file', nargs='?')
    args = parser.parse_args()

    if args.dataset_path is None:
        # Default should be 'tests/datasets/op_cog_dataset.json', however that is from project root.
        # Make sure it works anywhere by getting the path of this file and appending the default dataset path.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset_path = os.path.join(current_dir, '../../datasets/op_cog_dataset.json')

    discrepancies, avg_time_incl_first_1, avg_time_excl_first_1, avg_time_incl_first_2, avg_time_excl_first_2 = compare_csv(args.file1, args.file2, args.dataset_path)

    if discrepancies:
        print("Discrepancies found:")
        for discrepancy in discrepancies:
            print(discrepancy)
    else:
        print("No discrepancies found.")

    print(f"\nAverage execution time for {args.file1} (including first entry): {avg_time_incl_first_1:.5f} seconds")
    print(f"Average execution time for {args.file1} (excluding first entry): {avg_time_excl_first_1:.5f} seconds")
    print(f"\nAverage execution time for {args.file2} (including first entry): {avg_time_incl_first_2:.5f} seconds")
    print(f"Average execution time for {args.file2} (excluding first entry): {avg_time_excl_first_2:.5f} seconds")
