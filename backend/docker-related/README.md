# Creation

Create a .env file in the 'docker-related' directory (this directory) to include your environment variables. The .env.example file shows some variables you might want to add and how to format them.

To build the simulator container from scratch, run this command in the VISION backend root directory:

```
docker build -t simulator-container .
```

To run the simulator from this directory (to run from another directory update the path to .env). Specify the name of your container in place of <session-name>:

```
docker run --rm -it --privileged --name <session-name> -p 6380:6379 -v redis-data:/data --env-file .env simulator-container
```

Once the docker is run, all you need to do is start the VISION backend if you're in `.venv` (make sure the conda environment is inactive with `conda deactivate`):

`python src/hal_beam_com/cog_manager.py`

# Troubleshooting

## ENV variables not exporting correctly
If you notice that you're getting an S3 error, or other env key related issues, the quotation marks you're using might mess up the env export.
As a temporary fix I just export all the env keys manually using `export`.
