# VISION - Front-end

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
3. Create the venv (with name e.g. .venv) with the new Python version
    ```bash
    pyenv virtualenv 3.12.7 .venv
    ```
4. Activate your Python environment:
    ```bash
    pyenv activate .venv
    ```

## Setup continued
1. Install the `requirements.txt`: `pip install -r ./requirements.txt`
1. Execute `python ./UI/program/executable.py` with your Python of choice to launch the UI.

## Other requirements
- xdotool (only for beamline key insertion)
  ```bash
  sudo apt-get install xdotool
  ```
