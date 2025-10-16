import time
import os
import argparse
import json
from sklearn.metrics import f1_score
from base_test_framework import BaseTestFramework
from src.hal_beam_com.cogs.classifier_cog import invoke
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.utils import SystemPromptType, get_data_field, CogType
from datetime import datetime
from src.hal_beam_com.utils import CogType


class ClassifierCogTestFramework(BaseTestFramework):
    def __init__(self, dataset_path, results_path, system_prompt_type, system_prompt_path, base_model, num_runs=1):
        super().__init__(dataset_path, results_path, system_prompt_path, base_model, num_runs=num_runs)
        self.fieldnames = ['input', 'expected_output', 'generated_output', 'match', 'execution_time']

        self.expected_outputs = []
        self.generated_outputs = []

        if system_prompt_type not in SystemPromptType:
            raise ValueError(f"Invalid system_prompt_type: {system_prompt_type}. Must be one of {[e.value for e in SystemPromptType]}")
        
        # Set the system_prompt_type if valid
        self.system_prompt_type = system_prompt_type

        # Preload the model and do a warm-up request
        model = ModelManager.get_model(base_model)
        print("Performing warm-up request...")
        warmup_data = [{
            'beamline': '11BM',
            'text_input': 'analyze the data',
            'include_context_functions': 0,
            'only_text_input': 1
        }]
        invoke(warmup_data, base_model=self.base_model, finetuned=False, system_prompt_type=self.system_prompt_type, system_prompt_path=self.system_prompt_path)
        print("Warm-up complete")

    def test_entry(self, entry):
        input_data = entry['input']
        expected_output = entry['expected_output']

        data = [{
            'beamline': '11BM',
            'text_input': input_data,
            'include_context_functions': 0,
            'only_text_input': 1
        }]

        start_time = time.time()
        result = invoke(data, base_model=self.base_model, finetuned=False, system_prompt_type = self.system_prompt_type, system_prompt_path = self.system_prompt_path, testing = True)
        end_time = time.time()
        execution_time = end_time - start_time
        self.execution_times.append(execution_time)

        # TODO: This should not be indexed with an index, but it is how the code is set-up for now
        generated_output = result[0]['classifier_cog_output']
        generated_output = generated_output.strip()
        execution_time = end_time - start_time
        self.execution_times.append(execution_time)

        match = generated_output.strip().lower() == expected_output.strip().lower()

        self.expected_outputs.append(expected_output.strip().lower())
        self.generated_outputs.append(generated_output.strip().lower())

        result_data = {
            'input': input_data,
            'expected_output': expected_output,
            'generated_output': generated_output,
            'match': match,
            'execution_time': f"{execution_time:.5f} seconds"
        }
        
        self.write_result(result_data)

        if self.system_prompt is None:
            self.system_prompt = get_data_field(result, CogType.CLASSIFIER, "system_prompt")
        
        return match

    
    def run_tests(self, run_statistics_path=None):
        super().run_tests(run_statistics_path)

        f1 = f1_score(self.expected_outputs, self.generated_outputs, average='weighted')

        if run_statistics_path:
            with open(run_statistics_path, 'r') as f:
                run_statistics = json.load(f)
            run_statistics["f1_score"] = f1
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)

if __name__ == '__main__':

    cog = CogType.CLASSIFIER

    parser = argparse.ArgumentParser(description='Run classifier_cog tests.')
    parser.add_argument('--base_model', type=str, default='qwen2', help='Base model to use')
    parser.add_argument('--system_prompt_path', type=str, help='Path to the system prompt file')
    parser.add_argument('--system_prompt_type', type=str,
                        choices=[SystemPromptType.ONE_WORD_OUTPUT.value, SystemPromptType.ID_OUTPUT.value, SystemPromptType.LIST_OUTPUT.value],
                        help='Type of system prompt to test')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to run the experiment')
    args = parser.parse_args()

    args.system_prompt_type = SystemPromptType(args.system_prompt_type) if args.system_prompt_type else SystemPromptType.ONE_WORD_OUTPUT

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(base_dir, 'datasets', f'{cog.value}_cog_dataset.json')
    
    # Create a parent timestamp directory for all runs
    parent_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(base_dir, 'results', f'{cog.value}_cog', args.system_prompt_type.value, args.base_model, parent_timestamp)
    os.makedirs(base_results_dir, exist_ok=True)

    # Run multiple experiments
    for run in range(args.num_runs):
        run_dir = os.path.join(base_results_dir, f'run_{run}')
        results_file_path = os.path.join(run_dir, f'results_{cog.value}_cog_{args.system_prompt_type.value}.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        tester = ClassifierCogTestFramework(
            dataset_file_path,
            results_file_path,
            args.system_prompt_type,
            args.system_prompt_path,
            args.base_model,
            num_runs=args.num_runs
        )
        
        run_statistics_path = os.path.join(run_dir, 'run_statistics.json')
        tester.run_tests(run_statistics_path)

    # Aggregate results after all runs are complete
    tester.aggregate_run_statistics(base_results_dir)