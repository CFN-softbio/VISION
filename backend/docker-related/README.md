# Creation

Create a .env file in the 'docker-related' directory (this directory) to include your environment variables. The .env.example file shows some variables you might want to add and how to format them.

To build the simulator container from scratch, run this command in the VISION backend root directory:

## Step 1: Build the Container

```bash
docker build -t simulator-container .
```

## Step 2: Run the Container with Shared Communication Directory

The container needs a shared directory to communicate with the frontend UI running on your host machine. The setup is OS-agnostic and works on Windows, Mac, and Linux.

### Default Shared Directory Locations

The shared directory will be automatically created at:
- **Windows**: `%LOCALAPPDATA%\hal_vision\shared` (typically `C:\Users\YourName\AppData\Local\hal_vision\shared`)
- **Mac/Linux**: `~/.hal_vision/shared`

### Running the Container

**For Mac/Linux:**
```bash
# Create the shared directory
mkdir -p ~/.hal_vision/shared

docker run --rm -it --privileged --name <session-name> \
  -p 6380:6379 \
  -v redis-data:/data \
  -v ~/.hal_vision/shared:/root/.hal_vision/shared \
  --env-file .env \
  simulator-container
```

**For Windows (PowerShell):**
```powershell
# Create the shared directory
New-Item -ItemType Directory -Force -Path "${env:LOCALAPPDATA}\hal_vision\shared"

# Start your container
docker run --rm -it --privileged --name <session-name> `
  -p 6380:6379 `
  -v redis-data:/data `
  -v ${env:LOCALAPPDATA}\hal_vision\shared:/root/.hal_vision/shared `
  --env-file docker-related/.env `
  simulator-container
```

**For Windows (Command Prompt):**
```cmd
REM Create the shared directory
mkdir %LOCALAPPDATA%\hal_vision\shared

docker run --rm -it --privileged --name <session-name> ^
  -p 6380:6379 ^
  -v redis-data:/data ^
  -v %LOCALAPPDATA%\hal_vision\shared:/root/.hal_vision/shared ^
  --env-file .env ^
  simulator-container
```

### Custom Shared Directory (Optional)

If you want to use a different location for the shared directory:

1. Set the `HAL_COMM_DIR` environment variable on your host machine before running the frontend
2. Mount that location when starting the container

**Example with custom directory:**

```bash
# Mac/Linux
export HAL_COMM_DIR=/path/to/custom/shared
docker run --rm -it --privileged --name <session-name> \
  -p 6380:6379 \
  -v redis-data:/data \
  -v /path/to/custom/shared:/root/.hal_vision/shared \
  -e HAL_COMM_DIR=/root/.hal_vision/shared \
  --env-file .env \
  simulator-container
```

```powershell
# Windows PowerShell
$env:HAL_COMM_DIR="C:\path\to\custom\shared"
docker run --rm -it --privileged --name <session-name> `
  -p 6380:6379 `
  -v redis-data:/data `
  -v C:\path\to\custom\shared:/root/.hal_vision/shared `
  -e HAL_COMM_DIR=/root/.hal_vision/shared `
  --env-file .env `
  simulator-container
```

## Step 3: Start the Backend

Once the docker is running, start the VISION backend if you're in `.venv` (make sure the conda environment is inactive with `conda deactivate`):

```bash
python src/hal_beam_com/cog_manager.py
```

The backend will automatically use the shared directory for communication with the frontend UI.

## Running on Bare Metal (Without Docker)

The LocalCustomS3 implementation also works without Docker. Both the backend and frontend will automatically use the shared communication directory:

- **Windows**: `%LOCALAPPDATA%\hal_vision\shared`
- **Mac/Linux**: `~/.hal_vision/shared`

Simply run both processes on the same machine and they will communicate through this shared directory. No additional configuration is needed unless you want to use a custom location via the `HAL_COMM_DIR` environment variable.

## Running Persistence Session with Tmux

If docker container has not already been built, build the docker container as **Step 1**.

Before **Step 2**, create a `tmux` session:
```bash
tmux new -s <session_name>
```

Then run **Step 2** with the same `session_name` and **Step 3**.

To detach from the session:
```
Ctrl + b, then press d
```

The session will run in the background.

## Reconnect to the Tmux Session
```bash
tmux attach -t <session_name>
```

# Troubleshooting

## Shared Directory Not Found / No Client Directories

If the backend shows "Found 0 client directory(s)" continuously:

1. **Verify volume mount is working:**
   ```bash
   # Inside container
   bash /usr/local/bin/verify_shared_dir.sh
   ```

2. **Check from host machine:**
   - **Windows**: Open `%LOCALAPPDATA%\hal_vision\shared` in Explorer
   - **Mac/Linux**: Run `ls -la ~/.hal_vision/shared`
   
   You should see directories like `transmissions/experiment_YYYY-MM-DD/`

3. **Verify Docker volume mount:**
   - Check your `docker run` command includes the correct `-v` flag
   - For Windows: `-v ${env:LOCALAPPDATA}\hal_vision\shared:/root/.hal_vision/shared`
   - For Mac/Linux: `-v ~/.hal_vision/shared:/root/.hal_vision/shared`

4. **Check both processes use same experiment name:**
   - Backend logs show: `Experiment: experiment_YYYY-MM-DD`
   - Frontend should use the same date format
   - Both must use the same date (generated from current system date)

5. **Verify frontend is running and connected:**
   - Frontend should create `user_{client_id}` directory
   - Check host machine shared directory for these folders

## ENV Variables Not Exporting Correctly

If you notice that you're getting an S3 error, or other env key related issues, the quotation marks you're using might mess up the env export.
As a temporary fix, export all the env keys manually using `export`.

## Permission Issues (Linux/Mac)

If you encounter permission issues with the shared directory:

1. Ensure the directory exists and has proper permissions:
```bash
mkdir -p ~/.hal_vision/shared
chmod -R 777 ~/.hal_vision/shared
```

2. If running in rootless Docker mode, you may need to adjust the user mapping or run with `--user $(id -u):$(id -g)`

## Shared Directory Not Working (Windows)

On Windows, ensure:
1. Docker Desktop has file sharing enabled for the drive containing the shared directory
2. The path doesn't contain special characters
3. You're using the correct path format (backslashes for local paths, forward slashes in Docker commands)
4. If using a custom path, make sure it's properly escaped in PowerShell (use quotes)

## Frontend Can't Connect to Backend

Verify that:
1. Both frontend and backend are using the same shared directory
2. The `HAL_COMM_DIR` environment variable (if set) matches on both sides
3. The directory has write permissions for both processes
4. No firewall is blocking file access
5. For Docker: the volume mount is correct in the docker run command
6. Both use MultiClient queue classes (MultiClientQueue_AI on backend, MultiClientQueue_user on frontend)

## Checking Communication Directory

To verify which directory is being used, check the console output when starting either the backend or frontend. It will display:
```
Using shared communication directory: /path/to/directory
```

Make sure this path is the same for both processes.

## Debugging Communication Issues

If communication fails but directory setup looks correct:

1. **Check directory tree from within container:**
   ```bash
   # Inside container
   ls -R /root/.hal_vision/shared/
   ```

2. **Compare with host:**
   ```bash
   # On host (Mac/Linux)
   ls -R ~/.hal_vision/shared/
   
   # On host (Windows PowerShell)
   Get-ChildItem -Recurse ${env:LOCALAPPDATA}\hal_vision\shared
   ```

3. **Enable detailed logging:**
   - Backend logs include `[DEBUG BACKEND]` messages showing directory scans
   - Look for "Full directory tree" output every 100 scan attempts
   - Check for permission errors or "does not exist" messages

4. **Test file creation:**
   ```bash
   # Inside container
   touch /root/.hal_vision/shared/test.txt
   ```
   
   Then check if file appears on host machine

5. **Restart both processes:**
   - Sometimes a clean restart resolves stale state issues
   - Make sure to clear old experiment directories if needed
