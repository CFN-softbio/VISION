import time
import os
import argparse
import Levenshtein
import json
from datetime import datetime
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from codebleu import calc_codebleu

from src.hal_beam_com.model_manager import ModelManager
from utils import strip_comments
from base_test_framework import BaseTestFramework
from src.hal_beam_com.cogs.op_cog import invoke
from src.hal_beam_com.utils import get_data_field, CogType


class OpAgentTestFramework(BaseTestFramework):
    def __init__(self, dataset_path, results_path, base_model='mistral', system_prompt_path=None, num_runs=1):
        super().__init__(dataset_path, results_path, system_prompt_path=system_prompt_path, base_model=base_model, num_runs=num_runs)
        self.system_prompt_path = system_prompt_path
        self.fieldnames = (['command', 'expected_codes', 'generated_code', 'exact_match', 'execution_time', 
                         'codebleu_scores', 'levenshtein_distances', 'normalized_levenshtein_distances',
                         'best_normalized_levenshtein_distance', 'is_complex']
                         + list(calc_codebleu(["print('hello')"], ["print('hello')"], lang="python").keys())
                         + ['best_codebleu_score', 'best_levenshtein_distance', 'average_levenshtein_distance'])

        # Single dictionary to track all metrics by complexity
        self.metrics = {
            'simple': {'results': [], 'execution_times': [], 'count': 0, 'correct_matches': 0},
            'complex': {'results': [], 'execution_times': [], 'count': 0, 'correct_matches': 0}
        }

        # Preload the model and do a warm-up request
        model = ModelManager.get_model(base_model)
        print("Performing warm-up request...")
        warmup_data = [{
            'beamline': '11BM',
            'text_input': 'print hello',
            'include_context_functions': True,
            'only_text_input': 1
        }]
        invoke(warmup_data, base_model=self.base_model, finetuned=False, system_prompt_path=self.system_prompt_path)
        print("Warm-up complete")

    def test_entry(self, entry):
        command = entry['command']
        expected_codes = entry['expected_code']

        data = [{
            'beamline': '11BM',
            'text_input': command,
            'include_context_functions': True,
            'only_text_input': 1
        }]

        start_time = time.time()

        cog_result = invoke(data, base_model=self.base_model, finetuned=False, system_prompt_path=self.system_prompt_path)
        end_time = time.time()
        execution_time = end_time - start_time
        self.execution_times.append(execution_time)

        generated_code = cog_result[0][f'{CogType.OP.value}_cog_output']
        execution_time = end_time - start_time
        print(execution_time)

        # Strip comments before calculating metrics
        generated_code_clean = strip_comments(generated_code)
        expected_codes_clean = [strip_comments(code) for code in expected_codes]

        exact_match = any(generated_code.strip() == expected_code.strip() for expected_code in expected_codes_clean)

        codebleu_scores = [calc_codebleu([clean_expected], [generated_code_clean], lang="python",
                                         weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                           for clean_expected in expected_codes_clean]

        # Calculate Levenshtein distances using cleaned code
        levenshtein_distances = [Levenshtein.distance(generated_code_clean, clean_expected) 
                                for clean_expected in expected_codes_clean]

        # Calculate normalized Levenshtein distances
        normalized_levenshtein_distances = [
            dist / max(len(generated_code_clean), len(clean_expected))
            for dist, clean_expected in zip(levenshtein_distances, expected_codes_clean)
        ]

        # Find the best metrics
        best_codebleu_output = max(codebleu_scores, key=lambda x: x['codebleu'])
        best_levenshtein_distance = min(levenshtein_distances)
        best_normalized_levenshtein_distance = min(normalized_levenshtein_distances)

        # Determine complexity and store metrics
        is_complex = entry.get('is_complex', False)
        complexity_type = 'complex' if is_complex else 'simple'
        
        is_correct = exact_match or best_codebleu_output['codebleu'] == 1.0
        
        result = {
            'codebleu': best_codebleu_output,
            'levenshtein': best_levenshtein_distance,
            'normalized_levenshtein': best_normalized_levenshtein_distance
        }
        
        self.metrics[complexity_type]['results'].append(result)
        self.metrics[complexity_type]['execution_times'].append(execution_time)
        self.metrics[complexity_type]['count'] += 1
        self.metrics[complexity_type]['correct_matches'] += 1 if is_correct else 0

        result_data = {
            "best_levenshtein_distance": best_levenshtein_distance,
            "best_normalized_levenshtein_distance": best_normalized_levenshtein_distance,
            "command": command,
            "expected_codes": json.dumps(expected_codes),  # Store as JSON array
            "generated_code": generated_code,
            "exact_match": exact_match,
            "execution_time": f"{execution_time:.5f} seconds",
            "codebleu_scores": json.dumps(codebleu_scores),  # Store as JSON array
            "levenshtein_distances": json.dumps(levenshtein_distances),  # Store as JSON array
            "normalized_levenshtein_distances": json.dumps(normalized_levenshtein_distances),  # Store as JSON array
            "best_codebleu_score": best_codebleu_output
        }

        self.write_result(result_data)

        if self.system_prompt is None:
            self.system_prompt = get_data_field(cog_result, CogType.OP, "system_prompt")

        return exact_match or best_codebleu_output['codebleu'] == 1.0

    def calculate_statistics(self):
        stats = {}
        for complexity_type in ['simple', 'complex']:
            results = self.metrics[complexity_type]['results']
            if not results:
                continue
                
            count = self.metrics[complexity_type]['count']
            correct = self.metrics[complexity_type]['correct_matches']
            
            stats[complexity_type] = {
                'count': count,
                'correct_matches': correct,
                'accuracy': (correct / count * 100) if count > 0 else 0,
                'average_best_codebleu': {
                    key: sum(r['codebleu'][key] for r in results) / len(results)
                    for key in results[0]['codebleu'].keys()
                },
                'average_best_levenshtein': sum(r['levenshtein'] for r in results) / len(results),
                'average_best_normalized_levenshtein': sum(r['normalized_levenshtein'] for r in results) / len(results),
                'average_execution_time': sum(self.metrics[complexity_type]['execution_times']) / len(results)
            }
        return stats

    def run_tests(self, run_statistics_path=None):
        super().run_tests(run_statistics_path)
        
        # Calculate and print statistics
        stats = self.calculate_statistics()
        for complexity_type, metrics in stats.items():
            print(f"\n{complexity_type.title()} Commands Statistics:")
            print(f"Count: {metrics['count']}")
            print(f"Average CodeBLEU Scores: {metrics['average_best_codebleu']}")
            print(f"Average Levenshtein Distance: {metrics['average_best_levenshtein']:.4f}")
            print(f"Average Normalized Levenshtein: {metrics['average_best_normalized_levenshtein']:.4f}")
            print(f"Average Execution Time: {metrics['average_execution_time']:.4f}s")

        if run_statistics_path:
            with open(run_statistics_path, 'r') as f:
                run_statistics = json.load(f)
            run_statistics["metrics_by_complexity"] = stats
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OpAgent tests.')
    parser.add_argument('--base_model', type=str, default='claude-3.5-sonnet', help='Base model to use')
    parser.add_argument('--system_prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to run the experiment')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(base_dir, 'datasets', 'op_cog_dataset.json')
    
    # Create a parent timestamp directory for all runs
    parent_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(base_dir, 'results', 'op_cog', args.base_model, parent_timestamp)
    os.makedirs(base_results_dir, exist_ok=True)

    # Run multiple experiments
    for run in range(args.num_runs):
        run_dir = os.path.join(base_results_dir, f'run_{run}')
        results_file_path = os.path.join(run_dir, 'results_op_agent.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        tester = OpAgentTestFramework(
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
