import sys
import time
import os
import argparse
from collections import defaultdict
import Levenshtein
import json
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from src.hal_beam_com.model_manager import ModelManager
from utils import strip_comments
from base_test_framework import BaseTestFramework
from src.hal_beam_com.cogs.analysis_cog import invoke
from src.hal_beam_com.utils import get_data_field, CogType


class AnaAgentTestFramework(BaseTestFramework):
    def __init__(self, dataset_path, results_path, base_model='mistral', system_prompt_path=None, num_runs=1):
        super().__init__(dataset_path, results_path, system_prompt_path=system_prompt_path, base_model=base_model,
                         num_runs=num_runs)
        self.system_prompt_path = system_prompt_path
        self.fieldnames = (['command', 'expected_codes', 'generated_code', 'exact_match', 'execution_time', 'levenshtein_distances', 'normalized_levenshtein_distances',
                            'best_normalized_levenshtein_distance', 'best_levenshtein_distance'])

        self.best_codebleu_scores = defaultdict(list)
        self.best_levenshtein_distances = []
        self.best_normalized_levenshtein_distances = []

        # Preload the model and do a warm-up request
        model = ModelManager.get_model(base_model)
        print("Performing warm-up request...")
        warmup_data = [{
            'beamline': '11BM',
            'text_input': 'print hello',
            'include_context_functions': True,
            'only_text_input': 1
        }]
        invoke(warmup_data, base_model=self.base_model, finetuned=False, system_prompt_path=self.system_prompt_path, add_full_python_command=False)
        print("Warm-up complete")

    def test_entry(self, entry):
        command = entry['command']
        expected_codes = entry['expected_output']

        data = [{
            'beamline': '11BM',
            'text_input': command,
            'include_context_functions': True,
            'only_text_input': 1
        }]

        start_time = time.time()

        result = invoke(data, base_model=self.base_model, finetuned=False, system_prompt_path=self.system_prompt_path, add_full_python_command=False)
        end_time = time.time()
        execution_time = end_time - start_time
        self.execution_times.append(execution_time)

        generated_code = result[0][f'{CogType.ANA.value}_cog_output']
        execution_time = end_time - start_time
        print(execution_time)

        # Strip comments before calculating metrics
        generated_code_clean = strip_comments(generated_code)
        expected_codes_clean = [strip_comments(code) for code in expected_codes]

        exact_match = any(generated_code.strip() == expected_code.strip() for expected_code in expected_codes_clean)

        # Calculate Levenshtein distances using cleaned code
        levenshtein_distances = [Levenshtein.distance(generated_code_clean, clean_expected)
                                 for clean_expected in expected_codes_clean]

        # Calculate normalized Levenshtein distances
        normalized_levenshtein_distances = [
            dist / max(len(generated_code_clean), len(clean_expected))
            for dist, clean_expected in zip(levenshtein_distances, expected_codes_clean)
        ]

        best_levenshtein_distance = min(levenshtein_distances)
        best_normalized_levenshtein_distance = min(normalized_levenshtein_distances)

        self.best_levenshtein_distances.append(best_levenshtein_distance)
        self.best_normalized_levenshtein_distances.append(best_normalized_levenshtein_distance)

        result_data = {
            "best_levenshtein_distance": best_levenshtein_distance,
            "best_normalized_levenshtein_distance": best_normalized_levenshtein_distance,
            "command": command,
            "expected_codes": json.dumps(expected_codes),  # Store as JSON array
            "generated_code": generated_code,
            "exact_match": exact_match,
            "execution_time": f"{execution_time:.5f} seconds",
            "levenshtein_distances": json.dumps(levenshtein_distances),  # Store as JSON array
            "normalized_levenshtein_distances": json.dumps(normalized_levenshtein_distances),  # Store as JSON array
        }

        self.write_result(result_data)

        if self.system_prompt is None:
            self.system_prompt = get_data_field(result, CogType.ANA, "system_prompt")

        return exact_match

    def calculate_average_best_codebleu(self):
        average_codebleu = {key: sum(values) / len(values) for key, values in self.best_codebleu_scores.items()}
        return average_codebleu

    def run_tests(self, run_statistics_path=None):
        super().run_tests(run_statistics_path)

        average_best_levenshtein_distance = sum(self.best_levenshtein_distances) / len(self.best_levenshtein_distances)
        average_best_normalized_levenshtein = sum(self.best_normalized_levenshtein_distances) / len(
            self.best_normalized_levenshtein_distances)

        print(f"Average best Levenshtein Distance: {average_best_levenshtein_distance}")
        print(f"Average best Normalized Levenshtein Distance: {average_best_normalized_levenshtein:.4f}")

        if run_statistics_path:
            with open(run_statistics_path, 'r') as f:
                run_statistics = json.load(f)
            run_statistics["average_best_levenshtein_distance"] = average_best_levenshtein_distance
            run_statistics["average_best_normalized_levenshtein"] = average_best_normalized_levenshtein
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AnaAgent tests.')
    parser.add_argument('--base_model', type=str, default='mistral', help='Base model to use')
    parser.add_argument('--system_prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to run the experiment')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(base_dir, 'datasets', 'ana_cog_dataset.json')

    # Create a parent timestamp directory for all runs
    parent_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(base_dir, 'results', 'ana_cog', args.base_model, parent_timestamp)
    os.makedirs(base_results_dir, exist_ok=True)

    # Run multiple experiments
    for run in range(args.num_runs):
        run_dir = os.path.join(base_results_dir, f'run_{run}')
        results_file_path = os.path.join(run_dir, 'results_ana_agent.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        tester = AnaAgentTestFramework(
            dataset_file_path,
            results_file_path,
            base_model=args.base_model,
            system_prompt_path=args.system_prompt_path,
            num_runs=args.num_runs
        )

        run_statistics_path = os.path.join(run_dir, 'run_statistics.json')
        tester.run_tests(run_statistics_path)

    # Aggregate results after all runs are complete
    tester.aggregate_run_statistics(base_results_dir)
