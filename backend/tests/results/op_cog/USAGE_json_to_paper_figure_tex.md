# Using json_to_paper_figure_tex.py

This tool emits a LaTeX pgfplots data block (CSV inside `\pgfplotstableread{...}`) that you can `\input{...}` directly into your figure code.

It supports multiple "figure schemas" (named entries) with different columns and intended visualizations. You can select which models to include via `--include-keys` (in the exact order you want), or remove some via `--exclude-keys`.

Notes:
- Input can be either a `.json` model summary or the `.txt` version; the tool will auto-convert `.txt` using the existing converter.
- If you use a mapping file (recommended), pass it with `--mapping path/to/mapping.json` so your "display" names match your paper labels.
- If you pass `--include-keys`, those names select the models; ordering will be by full score unless you also pass `--keep-include-order` (to preserve your provided order). Labels in the CSV and in the "symbolic x coords" output always use the mapping's display names.
- The default table macro name is `\datatable`; change it with `--table-name '\\datatableSimple'` if you need different names per figure.

## First figure of the paper (SIMPLE flow data)

- Left bar: average full score (%)
- Right bar: stacked Exact Match (top, %) + Improvement (bottom, %) where Improvement = Functional − Exact
- Total column (Exact + Improvement) is emitted for convenience (used as the right-side number in your figure)

Run:

python tests/results/op_cog/json_to_paper_figure_tex.py /VISION_dev_be/tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.json --mapping tests/results/op_cog/model_table_mapping.json --complexity simple --figure simple_flow --include-keys "Claude 3.5 Sonnet (20241022)" "Claude Sonnet 4 (20250514, br)" "Claude Sonnet 4 (Abacus)" "o3 (high)" "Claude Opus 4 (Abacus)" "Claude Sonnet 4 (20250514)" "Claude Opus 4 (20250514)" "GPT-4o" "GPT-4o (Abacus)" "Grok-4 (Abacus)" "Gemini 2.5 Pro (Abacus)" "Athene-v2-Agent (72.7B)" "Athene-v2 (72.7B)" "Llama3.3 (70B)" "Qwen2.5 (7.62B)" "Qwen2 (7.62B)" "Qwen2.5-Coder (32.8B)" "Mistral-NeMo (12.2B)" "Mistral (7.25B)" "Phi-3.5 (3.8B, fp16)" "Phi-3.5 (3.8B)" "Qwen3 (32B)" "Qwen3 (32B, thinking)" --decimals 2 --output test.tex
This writes a snippet like:

% --- Data for SIMPLE FLOW (SIMPLE) ---
% SIMPLE-flow data (CSV; safer parsing)
\pgfplotstableread[col sep=semicolon]{ 
...CSV rows...
}\datatable

Include it in your LaTeX figure where your plotting code expects `\datatable`:

\input{tests/results/op_cog/simple_flow.simple.tex}

The script also outputs the ordered "symbolic x coords" list (printed to stdout and included as commented lines in the .tex). Copy it into your axis options, for example:
symbolic x coords={ Claude 3.5 Sonnet (20241022),Claude Sonnet 4 (20250514, br),Claude Sonnet 4 (Abacus),o3 (high),Claude Opus 4 (Abacus),Claude Sonnet 4 (20250514),Claude Opus 4 (20250514),GPT-4o,GPT-4o (Abacus),Grok-4 (Abacus),Gemini 2.5 Pro (Abacus),Athene-v2-Agent (72.7B),Athene-v2 (72.7B),Llama3.3 (70B),Qwen2.5 (7.62B),Qwen2 (7.62B),Qwen2.5-Coder (32.8B),Mistral-NeMo (12.2B),Mistral (7.25B),Phi-3.5 (3.8B, fp16),Phi-3.5 (3.8B),Qwen3 (32B),Qwen3 (32B, thinking) }

If you need a different macro name to avoid collisions, add `--table-name '\\datatableSimple'` and reference that in your TikZ/pgfplots code instead.

## Second figure of the paper (COMPLEX flow data)

- Left bar: average full score (%)
- Right bar: stacked Exact Match (top, %) + Improvement (bottom, %) where Improvement = Functional − Exact
- Total column (Exact + Improvement) is emitted for convenience (used as the right-side number in your figure)

Run:

python tests/results/op_cog/json_to_paper_figure_tex.py /VISION_dev_be/tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.json --mapping tests/results/op_cog/model_table_mapping.json --complexity complex --figure complex_flow --include-keys "Claude Sonnet 4 (20250514)" "Claude Opus 4 (20250514)" "Claude 3.5 Sonnet (20241022)" "GPT-4o" "o3 (high)" "GPT-4o (Abacus)" "Claude Opus 4 (Abacus)" "Claude Sonnet 4 (Abacus)" "Claude Sonnet 4 (20250514, br)" "Athene-v2 (72.7B)" "GPT-5 (high)" "Athene-v2-Agent (72.7B)" "Llama3.3 (70B)" "Qwen2.5 (7.62B)" "Qwen2.5-Coder (32.8B)" "Mistral-NeMo (12.2B)" "Qwen2 (7.62B)" "Gemini 2.5 Pro (Abacus)" "Mistral (7.25B)" "Qwen3 (32B)" "Qwen3 (32B, thinking)" "Phi-3.5 (3.8B)" "Phi-3.5 (3.8B, fp16)" --decimals 2 --output tests/results/op_cog/complex_flow.complex.tex

Symbolic x coords (ordered):
symbolic x coords={ {Claude Sonnet 4 (20250514)},{Claude Opus 4 (20250514)},{Claude 3.5 Sonnet (20241022)},{GPT-4o},{o3 (high)},{GPT-4o (Abacus)},{Claude Opus 4 (Abacus)},{Claude Sonnet 4 (Abacus)},{Claude Sonnet 4 (20250514, br)},{Athene-v2 (72.7B)},{GPT-5 (high)},{Athene-v2-Agent (72.7B)},{Llama3.3 (70B)},{Qwen2.5 (7.62B)},{Qwen2.5-Coder (32.8B)},{Mistral-NeMo (12.2B)},{Qwen2 (7.62B)},{Gemini 2.5 Pro (Abacus)},{Mistral (7.25B)},{Qwen3 (32B)},{Qwen3 (32B, thinking)},{Phi-3.5 (3.8B)},{Phi-3.5 (3.8B, fp16)} }

## Third figure of the paper (COMPLEX components)

- Bars: Full Score, PV match, Timing, Temp (%), sorted by Complex Full Score.
- Values are complex-only averages; Temp is computed only when applicable.
- Output uses comma-separated CSV and includes a first "model" column with the (escaped) model key.
- The tool looks for component metrics with keys: average_pv_match_rate, average_timing_score, average_temp_score (and also accepts common synonyms like pv_match, timing_score, temperature_score). Missing values default to 0.00 for this figure. Note: average_pv_match_rate is already expressed in percent, so it is not rescaled; timing and temp are scaled to percent if provided on 0-1 scale.

Recommended models (display names from mapping) to include, in order:

- Claude Sonnet 4 (20250514)
- Claude 3.5 Sonnet (20241022)
- GPT-4o
- GPT-5 (high)
- o3 (high)
- Llama3.3 (70B)
- Qwen2.5 (7.62B)
- Grok-4 (Abacus)

Run:

python tests/results/op_cog/json_to_paper_figure_tex.py tests/results/op_cog/saved_named_runs/new_prompt_new_dataset_12_8/model_comparison_summary.json --mapping tests/results/op_cog/model_table_mapping.json --complexity complex --figure complex_components --decimals 2 --include-keys "Claude Sonnet 4 (20250514)" "Claude 3.5 Sonnet (20241022)" "GPT-4o" "GPT-5 (high)" "o3 (high)" "Llama3.3 (70B)" "Qwen2.5 (7.62B)" "Grok-4 (Abacus)" --output tests/results/op_cog/complex_components.complex.tex

Example LaTeX snippet produced (structure):

\pgfplotstableread[col sep=comma]{%
model,label, fullscore, pvmatch, timing, temp
{claude-sonnet-4-20250514-thinking}, Claude Sonnet 4 (20250514), 90.73, 91.83, 85.55, 98.60
{claude-3.5-sonnet},                 Claude 3.5 Sonnet (20241022), 88.12, 88.51, 82.48, 97.59
{gpt-4o},                             GPT-4o,               85.04, 86.09, 80.63, 97.61
{gpt-5-high},                         GPT-5 (high),         85.25, 85.74, 82.42, 96.97
{o3-high},                            o3 (high),            84.71, 84.47, 80.61, 98.74
{grok-4-abacus},                      Grok-4 (Abacus),      17.88, 17.20, 21.32, 62.20
{llama3.3},                           Llama3.3 (70B),       49.50, 46.65, 65.23, 73.71
{qwen2.5},                            Qwen2.5 (7.62B),      47.48, 41.35, 72.92, 72.79
}\datatable

Copy the symbolic x coords (printed to stdout and included as comments in the .tex) into your axis options to preserve ordering.

## Legacy figure: old prompt + old dataset (SIMPLE flow)

- Bars: Same as the First figure (Average Full Score; Exact + Improvement = Functional).
- Input JSON: /home2/nvleuten/projects/VISION_dev_be/tests/results/op_cog/saved_named_runs/old_prompt_old_dataset_25_08/model_comparison_summary.json
- Uses comma-separated CSV via figure schema simple_flow_old.

Recommended models (display names from mapping) to include, in order:

- Claude 3.5 Sonnet (20241022)
- GPT-4o
- Athene-v2-Agent (72.7B)
- Athene-v2 (72.7B)
- Llama3.3 (70B)
- Qwen2.5-Coder (32.8B)
- Qwen2.5 (7.62B)
- Qwen2 (7.62B)
- Mistral-NeMo (12.2B)
- Mistral (7.25B)
- Phi-3.5 (3.8B)
- Phi-3.5 (3.8B, fp16)

Run:

python tests/results/op_cog/json_to_paper_figure_tex.py /home2/nvleuten/projects/VISION_dev_be/tests/results/op_cog/saved_named_runs/old_prompt_old_dataset_25_08/model_comparison_summary.json --mapping tests/results/op_cog/model_table_mapping.json --complexity simple --figure simple_flow_old --decimals 2 --include-keys "Claude 3.5 Sonnet (20241022)" "GPT-4o" "Athene-v2-Agent (72.7B)" "Athene-v2 (72.7B)" "Llama3.3 (70B)" "Qwen2.5-Coder (32.8B)" "Qwen2.5 (7.62B)" "Qwen2 (7.62B)" "Mistral-NeMo (12.2B)" "Mistral (7.25B)" "Phi-3.5 (3.8B)" "Phi-3.5 (3.8B, fp16)" --output tests/results/op_cog/simple_flow_old.simple.tex

## Other figures

- Use `--figure complex_flow --complexity complex` for a different schema with columns suitable for other visualizations.
- You can define additional named schemas by editing `DEFAULT_FIGURES` in `tests/results/op_cog/json_to_paper_figure_tex.py`.
- Use `--exclude-keys` to drop specific models, or `--include-keys` to whitelist exactly the ones (in the order) you want shown.

## Tips

- If your labels don’t match, pass a mapping JSON with a `"display"` field per model key so your `--include-keys` can use human-friendly names.
- The output file defaults to `<input>.<figure>.<complexity>.tex` if `--output` is not provided.
