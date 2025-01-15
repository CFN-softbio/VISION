# vision_hal
Repo for all processes on HAL

# Installation
## Installing Python
### Install Python at System Level
Take care when installing Python at the system level, as on Linux you have to make sure that it does not override the default `python3` as it can brick the system.
When on Linux, the [pyenv](#alternative-pyenv-manager-linux) package is recommended.
1. Download and install Python 3.12 (different versions might work, but mostly untested)
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
2. Set up environment variable `SECRET_S3_KEY`, this key is required for S3 communication.
   This application requires a secret key to be stored as an environment variable. Otherwise, it will default to `./S3_secret_key.txt` unless a path is provided in `./VISION_0/Model3.0/program/S3_test/CustomS3.py` 

   - **Windows:**
     1. Open Command Prompt or PowerShell.
     2. Run:
        ```cmd
        setx SECRET_S3_KEY "your_secret_key_value"
        ```
     3. You might have to restart your IDE for the changes to take effect.
   
   - **macOS/Linux:**
     1. Open Terminal.
     2. Add the following line to your shell configuration file (`~/.bashrc` or `~/.zshrc` for macOS/Linux):
        ```bash
        export SECRET_S3_KEY="your_secret_key_value"
        ```
     3. Apply the changes:
        ```bash
        source ~/.bashrc  # or source ~/.zshrc for zsh users
        ```
3. Run `python ./src/hal_beam_com/cog_manager.py` to launch the backend.