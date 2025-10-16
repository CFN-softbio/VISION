# VISION - Backend
Repo for all processes on our backend server (HAL)

# Installation
## Installing Python
### Install Python at System Level
Take care when installing Python at the system level, as on Linux you have to make sure that it does not override the default `python3` as it can brick the system.
When on Linux, the [pyenv](#alternative-pyenv-manager-linux) package is recommended.
1. Download and install Python 3.12.7 (different versions might work, but mostly untested)
2. Create a virtual environment in the root folder using your desired Python version.
   ```bash
   python -m venv .venv  # you might have to use `python3.12` instead of `python`
   ```
3. Activate the virtual environment  
   - **Linux/MacOS:**  
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**  
     ```bash
     .\.venv\Scripts\Activate.ps1  # or .\.venv\Scripts\activate.bat
     ```
### Alternative pyenv Manager (Linux)
1. Install pyenv: 
   ```bash
   curl https://pyenv.run | bash
   ```
   Add the following to `~/.bash_profile`:
    ```bash
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    # Source the .bashrc file
    if [ -f ~/.bashrc ]; then
        . ~/.bashrc
    fi
   ```
2. Install the specified Python version via pyenv
    ```bash
    pyenv install 3.12.7
    ```
3. Create the venv with the new Python version
    ```bash
    pyenv virtualenv 3.12.7 .venv
    ```
4. Activate your Python environment:
    ```bash
    pyenv activate .venv
    ```

## Setup continued
1. Install the `requirements.txt`: `pip install -r ./requirements.txt`
2. Set up environment variable `ANTHROPIC_API_KEY`, this key is required for calling the default selected language models. This application requires a secret key to be stored as an environment variable.  

   - **Windows:**
     1. Open Command Prompt or PowerShell.
     2. Run:
        ```cmd
        setx ANTHROPIC_API_KEY "your_secret_key_value"
        ```
     3. You might have to restart your IDE for the changes to take effect.
   
   - **macOS/Linux:**
     1. Open Terminal.
     2. Add the following line to your shell configuration file (`~/.bashrc` or `~/.zshrc` for macOS/Linux):
        ```bash
        export ANTHROPIC_API_KEY="your_secret_key_value"
        ```
     3. Apply the changes:
        ```bash
        source ~/.bashrc  # or source ~/.zshrc for zsh users
        ```
3. Run `python ./src/hal_beam_com/cog_manager.py` to launch the backend.

# Testing

In `/tests` you'll find relevant testing code for the several cogs. We have created datasets for some of the beamlines
that we deploy to. Testing our cogs will help us give confidence in their performance, as well as be able to structurally
improve the system.

For more information, visit the `README_TESTING.md` file.

## Optional – CMS Beamline Simulator

Running the automated *OP-Cog* tests (see `tests/test_op_cog.py`) properly requires the
beamline simulator. The simulator lives in two **Git sub-repositories** that
are already referenced in `.gitmodules`.

If you do **not** intend to run the simulator you can ignore this section.

### Clone with the submodules

```bash
# fresh clone (recommended)
git clone --recurse-submodules <this_repo_name>.git
cd <this_repo_name>

# or, if you already cloned without submodules
git submodule update --init --depth 1       # fetch the recorded commits
```

The sub-repos will appear here:

```
external/profile_collection/
external/example-services/
```

For complete installation and run instructions open  
`README_SIMULATOR.md`.

## EnvTrace package (new)

This repository now includes a domain-agnostic package, EnvTrace, for execution-trace alignment and semantic evaluation.

- Install in editable mode:
  ```bash
  pip install -e .
  ```
- Run the CLI to align two traces and compute scores:
  ```bash
  envtrace align --gt path/to/gt_trace.json --pred path/to/pred_trace.json --out results/envtrace_report.json
  ```

Trace JSON schema:
- Either a top-level object with an `events` array, or a raw list of events.
- Each event: `{"channel": "<name>", "timestamp": <float>, "value": <any>, "meta": {}}`.

Example minimal JSON:
```json
{
  "events": [
    {"channel": "motor:x", "timestamp": 0.00, "value": 0.0},
    {"channel": "det:Acquire", "timestamp": 0.10, "value": 1},
    {"channel": "det:Acquire", "timestamp": 1.10, "value": 0}
  ]
}
```

EnvTrace returns:
- Alignment between ground-truth and predicted traces,
- Discrete match metrics (match rate and exactness),
- Timing metrics (R², slope, duration ratio, interval MAPE),
- A weighted full score and a strict binary decision.

Quickstart:
- Check version: `envtrace version`
- Try with the included example traces:
  - Ground truth: `examples/traces/gt.json`
  - Predicted:    `examples/traces/pred.json`
  - Run: `envtrace align --gt examples/traces/gt.json --pred examples/traces/pred.json --out results/envtrace_report.json`

Developer tests:
- Core unit tests are under `tests/envtrace/`. Run them with your test runner (e.g., `pytest`).
