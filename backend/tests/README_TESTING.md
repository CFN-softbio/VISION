# Testing

## Operator Cog
### Testing your ground truths
**Requires simulator for good results**

If you want to evaluate if you're tracking the correct PVs when evaluating pieces of code in the simulator. It is wise to run it in debug mode, this turns off exact matching, such that all codes are evaluated using the simulator.
It will select the first ground truth as the predicted code.

`python tests/test_op_cog.py --debug --no-cache --no-gt-cache`

### Execute LLM code
**Requires simulator for good results**

```
python tests/test_op_cog.py --base_model gpt-4o --num_runs 3 --dataset_path tests/datasets/op_cog_dataset.json --system_prompt_path src/hal_beam_com/cogs/prompt_templates/archive/first_paper_operator_cog_prompt.txt
```

Runs gpt-4o three times with the new dataset using the old prompt.

### Test all models to generate comparison script (sequentially)
**Requires simulator for good results**

```
python tests/results/run_all_models_comparison.py --agent_type op --num_runs 3
```

```
python tests/results/run_all_models_comparison.py --agent_type op --num_runs 3 --complex_only
```

### Run debug mode (test simulator accuracy)
**Requires simulator in general**

`python tests/test_op_cog.py --debug --num_runs 3 --no-cache`

### Execute beamline scientist code
**Requires simulator for good results**

```
python tests/test_op_cog.py --base_model human --scientist_file tests/datasets/op_cog_beamline_scientists/bs_1.toml --complex_only
```

### Errors or FAQ
Please refer to `README_SIMULATOR.md` for errors or bugs when executing `test_op_cog.py`.