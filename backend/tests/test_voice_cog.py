import time
import os
import argparse
from collections import defaultdict
import json
from datetime import datetime

from base_test_framework import BaseTestFramework
from src.hal_beam_com.cogs.voice_cog import invoke
from src.hal_beam_com.utils import CogType

from evaluate import load
import numpy as np

class VoiceAgentTestFramework(BaseTestFramework):
    def __init__(self, dataset_path, results_path, word, finetuned, base_model='whisper-large-v3'):
        super().__init__(dataset_path, results_path, base_model=base_model)
        self.fieldnames = ['expected_transcription', 'generated_transcription', 'execution_time', 'word_error_rate']
        
        self.finetuned = finetuned
        self.word = word

    def run_tests(self, word, i = None, run_statistics_path=None):
        self.load_dataset()
        self.prepare_csv()
        correct_matches = 0
        total_entries = 0

        self.test_entry(self.dataset, i)
        self.close_csv()

        if run_statistics_path:
            run_statistics = {
                "total_entries": total_entries,
                "correct_matches": correct_matches,
            }
            with open(run_statistics_path, 'w') as f:
                json.dump(run_statistics, f, indent=4)

        if self.system_prompt:
            # Save the system prompt to a file
            system_prompt_path = os.path.join(os.path.dirname(self.results_path), 'system_prompt.txt')
            with open(system_prompt_path, 'w') as f:
                f.write(self.system_prompt)

    def test_entry(self, entry, i = None):
        
        data = [{
            'beamline': '11BM',
            'include_context_functions': False,
            'only_text_input': 0
        }]

        results, match = invoke(data, base_model=self.base_model, model_number = i, word = self.word,finetuned=self.finetuned, audio_path = 'test', dataset = self.dataset)

        for result_data in results:
            self.write_result(result_data)

        return match


if __name__ == '__main__':
    cog = CogType.VOICE

    parser = argparse.ArgumentParser(description=f'Run {cog.value} Agent tests.')
    parser.add_argument('--base_model', type=str, default='whisper-large-v3-most-recent', help='Base model to use')
    parser.add_argument('--word', type=str, default='all', help='Word to evalute ASR')

    parser.add_argument('--finetuned', type=bool, default=False, help='Whether to use finetuned model or not')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_file_path = os.path.join(base_dir, 'datasets', f'{cog.value}_agent_dataset.json')

    num_finetuned_models = 238

    for i in range(num_finetuned_models):
        results_file_path = os.path.join(base_dir, 'results_paper_voice', f'{cog.value}_agent', args.base_model, args.word, str(i), f'results_{cog.value}_cog.csv')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        tester = VoiceAgentTestFramework(dataset_file_path, results_file_path, args.word, args.finetuned, args.base_model)

        run_statistics_path = os.path.join(base_dir, 'results_paper_voice', f'{cog.value}_agent', args.base_model, args.word, str(i), 'run_statistics.json')

