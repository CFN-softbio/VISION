# Beamline t01 IOC Instances and Services

This repository holds the a definition of example beamline t01 IOC Instances and services. It is a example of how to deploy epics-containers IOCs using docker compose for those facilities that are not using Kubernetes. It can also deploy its set of IOCs to a developer workstation for testing / experimentation.

The top level compose.yml file represents a set of IOCs and other services that would be deployed to a single IOC server.

For this example we have a single compose file. However, if you wanted to keep all IOCs for a beamline in a single repo but deploy to multiple servers, then each server would have its own named compose file.

## Initial Setup

First install Docker and Docker Compose. See https://docs.docker.com/compose/install/.

At DLS you need only run `module load docker-compose` to enable `docker compose` backed by the podman container engine. (see the end of this page if you get errors)

Setup command line completion for docker compose (optional). Note - using the short alias `dc` for `docker compose` will most likely not work with auto completion in bash. Using zsh instead of bash will give you completion even with aliasing.

Command line completion is very helpful as it will help you find the correct service names and commands.

Command line completion for docker:
```bash
# these steps will make cli completion work for zsh
mkdir -p ~/.oh-my-zsh/completions
docker completion zsh > ~/.oh-my-zsh/completions/_docker

# these steps will make cli completion work for bash
mkdir -p ~/.local/share/bash-completion/completions
docker completion bash > ~/.local/share/bash-completion/completions/docker
```

Command line completion for podman:
```bash
# these steps will make cli completion work for zsh
mkdir -p ~/.oh-my-zsh/completions
podman completion zsh > ~/.oh-my-zsh/completions/_podman

# these steps will make cli completion work for bash
mkdir -p ~/.local/share/bash-completion/completions
podman completion bash > ~/.local/share/bash-completion/completions/podman
```

## Local Testing Environment

To launch a test environment on a workstation, including phoebus perform the following steps:

```bash
git clone https://github.com/epics-containers/example-services.git
cd example-services
source ./environment.sh
docker compose up -d
```

NOTE: -d detaches from the containers. You may omit this if you would prefer to follow the logs of all the containers - these combined logs include a colour coded prefix to make them more legible.

This will launch the following containers:
- ca-gateway
- phoebus
- a motor simulation IOC
- an area detector simulation IOC
- an additional simple example IOC


## Experimenting
You can now try the following (we use `dc` as a short alias for `docker compose`):

```bash
# use caget/put locally
export EPICS_CA_ADDR_LIST=127.0.0.1:5094
caget BL01T-DI-CAM-01:DET:Acquire_RBV

# OR if you don't have caget/put locally then use one of the containers instead:
docker compose exec bl01t-ea-test-01 bash
export EPICS_CA_ADDR_LIST=127.0.0.1:5094
caget BL01T-DI-CAM-01:DET:Acquire_RBV

# attach to logs of a service (-f follows the logs, use ctrl-c to exit)
docker compose logs bl01t-di-cam-01 -f
# stop a service
docker compose stop bl01t-di-cam-01
# restart a service
docker compose start bl01t-di-cam-01
# attach to a service stdio
docker compose attach bl01t-di-cam-01
# exec a process in a service
docker compose exec bl01t-di-cam-01 bash
# delete a service (deletes the container)
docker compose down bl01t-di-cam-01
# create and launch a single service (plus its dependencies)
docker compose up bl01t-di-cam-01 -d
# close down and delete all the containers
# volumes are not deleted to preserve the data
docker compose down
```

# Deploy To Beamline Servers

TODO: this is work in progress - we have not yet created appropriate ca-gateway settings for this.

To deploy IOCs to a server, clone this repo and run the following command from the repo root:

```bash
docker compose --profile deploy up -d
```

or for a multiple server repo:
```bash
docker compose --profile deploy -f my_server_01.yml up -d
```

IMPORTANT: if you are using docker then IOCs deployed this way will automatically be brought up again on server reboot. podman will not do this by default because it is running in user space - there are workarounds for this but podman is not recommended for this purpose.

The gold standard for orchestrating these containers in production is Kubernetes. See https://epics-containers.github.io/main/tutorials/setup_k8s.html. Although compose is really useful for development and testing, Kubernetes is far superior for managing services across a cluster of hosts.

# Compose goals

These goals for switching to compose (from bespoke code in the `ec` tool) have all been met:

- be as DRY as possible
- work with docker-compose controlling either docker or podman
- enable isolated testing where PVs are not available to the whole subnet
- include separate profiles for:
  - local testing - including phoebus OPI
  - deployment to a beamline server - this would need either:
    - network host on the IOCs
    - a ca-gateway
- structure so that there is a compose file per server
- remove need for custom code/scripts to deploy/manage the IOCs
- also allow PV isolation on servers with a ca-gateway to enable access

# DLS Troubleshooting for docker compose module

Some users who have set up podman sockets in the past may get errors with `module load docker-compose`. If you do then do the following and then re-run it.
```bash
/dls_sw/apps/setup-podman/setup.sh
sed -i ~/.config/containers/containers.conf -e '/label=false/d' -e '/^\[containers\]$/a label=false'
```
