import json
import csv
import os
import sys
import json
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

class BaseTestFramework:
    def __init__(self, dataset_path, results_path, system_prompt_path=None, base_model=None, num_runs=1):
        self.fieldnames = None
        self.writer = None
        self.dataset = None
        self.csvfile = None
        self.execution_times = []
        
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.system_prompt_path = system_prompt_path

        self.base_model = base_model

        self.system_prompt = None  # Optional

        self.execution_times = []

        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)

    def prepare_csv(self):
        self.csvfile = open(self.results_path, 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write_result(self, result):
        self.writer.writerow(result)

    def close_csv(self):
        self.csvfile.close()

    def run_tests(self, run_statistics_path=None):
        self.load_dataset()
        self.prepare_csv()
        correct_matches = 0
        total_entries = len(self.dataset)
        
        for entry in self.dataset:
            if self.test_entry(entry):
                correct_matches += 1
        
        self.close_csv()
        
        accuracy = (correct_matches / total_entries) * 100
        print(f"\nFinal Accuracy: {accuracy:.2f}%")

        avg_execution_time = np.mean(self.execution_times)

        if run_statistics_path:
            run_statistics = {
                "accuracy": accuracy,
                "total_entries": total_entries,
                "correct_matches": correct_matches,
                "average_execution_time": avg_execution_time
            }
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)

        if self.system_prompt:
            # Save the system prompt to a file
            system_prompt_path = os.path.join(os.path.dirname(self.results_path), 'system_prompt.txt')
            with open(system_prompt_path, 'w') as f:
                f.write(self.system_prompt)

    def test_entry(self, entry):
        raise NotImplementedError("Subclasses should implement this method")

    def aggregate_run_statistics(self, base_results_dir):
        """Aggregate statistics from multiple runs into a single JSON file."""
        all_stats = []
        
        # Collect all run_statistics.json files
        for stats_file in Path(base_results_dir).glob('*/run_statistics.json'):
            with open(stats_file, 'r') as f:
                all_stats.append(json.load(f))
        
        # Merge all statistics
        merged_stats = {}
        for key in all_stats[0].keys():
            merged_stats[key] = [run[key] for run in all_stats]
        
        # Save aggregated results
        aggregated_file = os.path.join(base_results_dir, 'aggregated_statistics.json')
        with open(aggregated_file, 'w') as f:
            json.dump(merged_stats, f, indent=4)
