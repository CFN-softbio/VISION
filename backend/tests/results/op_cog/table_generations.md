Start all these with:
`python ./tests/results/op_cog/json_to_paper_table_csv.py`

# Table 1: New Dataset, New Prompt: simple
```
./tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.json --mapping model_table_mapping.json --sort-by avg_full_score_mean --desc --latex-rows test.txt --complexity simple
```

# Table 2: New Dataset, New Prompt: complex
```
./tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.json --mapping model_table_mapping.json --sort-by avg_full_score_mean --desc --latex-rows test.txt --complexity complex
```

# SI Table: Old Dataset, Old Prompt: simple
```
./tests/results/op_cog/saved_named_runs/old_prompt_old_dataset_25_08/model_comparison_summary.json --mapping model_table_mapping.json --sort-by avg_full_score_mean --desc --latex-rows test.txt --complexity simple
```