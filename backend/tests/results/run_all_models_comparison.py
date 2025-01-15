import os
import subprocess
from datetime import datetime
from pathlib import Path
import argparse
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from src.hal_beam_com.utils import SystemPromptType, base_models_path

def run_tests_for_all_models(agent_type, num_runs = '5', system_prompt_type = None):
    # Store paths to all aggregated statistics files
    model_results = {}

    # Run tests for each model
    for model_name in base_models_path.keys():
        print(f"\n{'='*50}")
        print(f"Running tests for model: {model_name}")
        print(f"{'='*50}\n")

        # Run the test script for this model
        # Get the path relative to this script (going up one level from results/ to tests/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_script = os.path.join(os.path.dirname(current_dir),
                                 f"test_{agent_type}_agent.py")
        
        cmd = [
            "python3",
            test_script,
            "--base_model", model_name,
            "--num_runs", num_runs,
             # You can adjust the number of runs
        ] 

        if system_prompt_type is not None:
            cmd += ["--system_prompt_type", system_prompt_type.value]

        try:
            subprocess.run(cmd, check=True)
            
            # Get the path relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))

            if system_prompt_type is not None:
                model_results_dir = Path(current_dir) / f"{agent_type}_agent/{system_prompt_type.value}" / model_name

            else:
                model_results_dir = Path(current_dir) / f"{agent_type}_agent" / model_name

            if model_results_dir.exists():
                latest_run = max(model_results_dir.iterdir(), key=os.path.getctime)
                stats_file = latest_run / "aggregated_statistics.json"
            
                if stats_file.exists():
                    model_results[model_name] = stats_file

        except subprocess.CalledProcessError as e:
            print(f"Error running tests for {model_name}: {e}")
            continue

    return model_results

def create_combined_summary(model_results, agent_type, system_prompt_type = None):
    # Get the path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if system_prompt_type is not None:
        summary_file = Path(current_dir) / f"{agent_type}_agent/{system_prompt_type.value}" / "model_comparison_summary.txt"

    else:
        summary_file = Path(current_dir) / f"{agent_type}_agent" / "model_comparison_summary.txt"
    
    with open(summary_file, "w") as f:
        f.write(f"{agent_type.upper()} Agent Model Comparison Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for model_name, stats_file in model_results.items():
            # Run analyze_statistics.py for this model's results
            # Get the path relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            analyze_script = os.path.join(current_dir, "analyze_statistics.py")
            
            cmd = [
                "python3",
                analyze_script,
                str(stats_file),
                "--agent_type", agent_type
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error analyzing statistics for {model_name}: {e}")
                continue

            f.write(f"\n{model_name}\n")
            f.write("-"*len(model_name) + "\n")
            
            # Copy the content of the model's statistics_summary.txt
            summary_path = stats_file.parent / "statistics_summary.txt"
            if summary_path.exists():
                with open(summary_path, "r") as model_summary:
                    f.write(model_summary.read() + "\n")
            f.write("\n" + "="*80 + "\n")

    print(f"\nComparison complete! Summary available at: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Run comparison tests across all models for a specific agent.')
    parser.add_argument('--agent_type', type=str, required=True,
                  choices=['op', 'ana', 'classifier'],
                  help='Type of agent to test')

    # Only add system_prompt_type argument if agent_type is classifier
    parser.add_argument('--system_prompt_type', type=str,
                    choices=[prompt_type.value for prompt_type in SystemPromptType],
                    help='Type of system prompt to test')
    
    parser.add_argument('--num_runs', type=str, default = '5',
                  help='Number of runs for the models')
    
    args = parser.parse_args()

    # Convert string to enum if system_prompt_type is provided
    if args.agent_type == 'classifier':
        args.system_prompt_type = SystemPromptType(args.system_prompt_type)
        prompt_info = f" with system prompt type {args.system_prompt_type}"
    else:
        prompt_info = ""

    print(f"Starting model comparison tests for {args.agent_type} agent{prompt_info}...")
    model_results = run_tests_for_all_models(
        args.agent_type, 
        args.num_runs, 
        system_prompt_type=getattr(args, 'system_prompt_type', None)
    )

    if model_results:
        print("\nAnalyzing results...")
        create_combined_summary(
            model_results, 
            args.agent_type, 
            system_prompt_type=getattr(args, 'system_prompt_type', None)
        )   
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()
