# Creation

Create a .env file in the 'docker-related' directory (this directory) to include your environment variables. The .env.example file shows some variables you might want to add and how to format them.

To build the simulator container from scratch, run this command in the VISION backend root directory:

<b>Step 1</b>
```
docker build -t simulator-container .
```

To run the simulator from this directory (to run from another directory update the path to .env). Specify the name of your container in place of <session-name>:

<b>Step 2</b>
```
docker run --rm -it --privileged --name <session-name> -p 6380:6379 -v redis-data:/data --env-file .env simulator-container
```

Once the docker is run, all you need to do is start the VISION backend if you're in `.venv` (make sure the conda environment is inactive with `conda deactivate`):

<b>Step 3</b>
`python src/hal_beam_com/cog_manager.py`

## Running persistence session with Tmux

If docker container has not already built, build the docker container as <b>Step 1</b>

Before <b>Step 2</b> Create `tmux` session
`tmux new -s <session_name>`

Then run <b>Step 2</b> with the same `session_name` and <b>Step 3</b>

Then,
`Ctrl + b` then press `d`

The session will run in the backgound.

## Reconnect to the Tmux session
`tmux attach -t <session_name>`

# Troubleshooting

## ENV variables not exporting correctly
If you notice that you're getting an S3 error, or other env key related issues, the quotation marks you're using might mess up the env export.
As a temporary fix I just export all the env keys manually using `export`.
