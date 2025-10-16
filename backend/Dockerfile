FROM ubuntu:22.04

# Install required tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    git \
    gnupg gnupg1 gnupg2 coreutils redis redis-server build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY . /home/vision/

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Simulator Conda Environment
RUN mkdir -p /opt/conda/envs/2024-3.0-py311-tiled && \
wget -P /opt/conda/envs/2024-3.0-py311-tiled https://zenodo.org/records/14019710/files/2024-3.0-py311-tiled.tar.gz && \
tar -xvf /opt/conda/envs/2024-3.0-py311-tiled/2024-3.0-py311-tiled.tar.gz -C /opt/conda/envs/2024-3.0-py311-tiled

# Redis setup
## Create a data dir for persistence and a minimal config
RUN mkdir -p /usr/local/etc/redis /data

## Minimal config suitable for containers:
## - daemonize no  -> keep redis in the foreground (PID 1)
## - bind 0.0.0.0 / protected-mode no only if you need external access
##   (safer to keep default bind 127.0.0.1 when used in the same container)
ADD docker-related/redis.conf /usr/local/etc/redis/redis.conf

VOLUME ["/data"]
EXPOSE 6379

# Install Docker CLI (no daemon)
## - Uses /etc/apt/keyrings (no apt-key)
## - Uses VERSION_CODENAME from /etc/os-release (no lsb_release)
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      -o /etc/apt/keyrings/docker.asc && \
    mkdir -p /etc/apt/sources.list.d; \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y docker-ce-cli docker-ce containerd.io docker-buildx-plugin docker-compose-plugin

# Install MongoDB
## Build-time: no interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor && \ 
   touch /etc/apt/sources.list.d/mongodb-org-8.0.list && \ 
   echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" > /etc/apt/sources.list.d/mongodb-org-8.0.list && apt-get update && apt-get install -y mongodb-org

# Setup Pyenv
## System deps (trim as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev uuid-dev tk-dev wget xz-utils \
  && rm -rf /var/lib/apt/lists/*

## pyenv and pyenv-virtualenv
ENV PYENV_ROOT=/opt/pyenv
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
RUN git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT" && \
    git clone https://github.com/pyenv/pyenv-virtualenv.git \
        "$PYENV_ROOT/plugins/pyenv-virtualenv"

## Install Python(s)
ARG PYTHON_VERSION=3.12.7
RUN bash -lc "eval \"\$(pyenv init -)\" && eval \"\$(pyenv virtualenv-init -)\" && \
    pyenv install -s ${PYTHON_VERSION} && \
    pyenv virtualenv ${PYTHON_VERSION} .venv && \
    pyenv global .venv"

# Add Entry Script
COPY docker-related/start.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh 

# Default to bash
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
