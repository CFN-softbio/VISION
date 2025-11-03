#! /bin/bash

# Setup environment variables required to launch the services described in this
# repo. A standard install of docker compose and permission to run docker
# are the only other requirements (membership of the docker group).
#
# docker compose may be backed by podman or docker container engines, see
# https://epics-containers.github.io/main/tutorials/setup_workstation.html.

# This script must be sourced
if [ "$0" = "$BASH_SOURCE" ]; then
    echo "ERROR: Please source this script (source ./environment.sh)"
    exit 1
fi

# if there is a docker-compose module then load it
if [[ $(module avail docker-compose 2>/dev/null) != "" ]] ; then
    module load docker-compose
fi

# podman vs docker differences.
if podman version &> /dev/null && [[ -z $USE_DOCKER ]] ; then
    USER_ID=0; USER_GID=0
    DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
    docker=podman
else
    USER_ID=$(id -u); USER_GID=$(id -g)
    unset DOCKER_HOST
    docker=docker
fi
echo using $docker as container engine

# ensure local container users can access X11 server
xhost +SI:localuser:$(id -un)

# Set up the environment for compose ###########################################

# set user id for the phoebus container for easy X11 forwarding.
export UIDGID=$USER_ID:$USER_GID
# default to the test profile for docker compose
export COMPOSE_PROFILES=test
# for test profile our ca-gateway publishes PVS on the loopback interface
export EPICS_CA_ADDR_LIST="127.0.0.1:5094 127.0.0.1"
# make a short alias for docker-compose for convenience
alias dc='$docker compose'

# Per https://stackoverflow.com/questions/72586838/xquartz-cant-open-display-mac-os:
export DISPLAY="$(/usr/sbin/ipconfig getifaddr en0):0"
