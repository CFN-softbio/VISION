#!/bin/sh

# Source Script for Simulator
source vision/external/example-services/environment.sh

# Manually Start Docker Daemon
dockerd --host=unix:///var/run/docker.sock >/var/log/dockerd.log 2>&1 &

# Start redis service
nohup redis-server /usr/local/etc/redis/redis.conf &


# Initialize Miniconda and Simulator Environment
conda init
echo "conda deactivate" >> ~/.bashrc
echo "conda tos accept" >> ~/.bashrc
echo "conda activate 2024-3.0-py311-tiled" >> ~/.bashrc
echo "conda-unpack" >> ~/.bashrc
echo "conda deactivate" >> ~/.bashrc

# Set up simulator
echo "conda activate 2024-3.0-py311-tiled" >> ~/.bashrc
echo "source ./vision/external/example-services/environment.sh" >> ~/.bashrc
echo "echo 'Starting Spoof Beamline'" >> ~/.bashrc
echo "nohup python ./vision/external/profile_collection/iocs/spoof_beamline.py &" >> ~/.bashrc

# TODO: I think you can remove these 3 again
# echo "cd ./vision/external/profile_collection" >> ~/.bashrc
# echo "./.ci/apply-autosettings.sh" >> ~/.bashrc
# echo "ipython --profile-dir=. --no-term-title --simple-prompt" >> ~/.bashrc

# Create pyenv for backend
# TODO: Please make this work
# apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
#   libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
#   xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
# curl -fsSL https://pyenv.run | bash
# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
# echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
# 
# echo "pyenv install 3.12.7" >> ~/.bashrc
# echo "pyenv virtualenv 3.12.7 .venv" >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
echo "pyenv activate .venv" >> ~/.bashrc
echo "cd /home/vision" >> ~/.bashrc
echo "pip install -r requirements.txt" >> ~/.bashrc

# Kafka Config
mkdir -p /etc/bluesky
tee /etc/bluesky/kafka.yml >/dev/null <<'EOF'
---
abort_run_on_kafka_exception: false
bootstrap_servers:
  - localhost:9092
runengine_producer_config:
  security.protocol: PLAINTEXT
EOF

# Other Setup
echo "127.0.0.1 info.cms.nsls2.bnl.gov" >> /etc/hosts

## Add cms profile
mkdir -p /root/.config/tiled/profiles
touch profiles.yml
tee /root/.config/tiled/profiles/profiles.yml >/dev/null <<'EOF'
cms:
  direct:
    authentication:
      allow_anonymous_access: true
    trees:
    - tree: databroker.mongo_normalized:Tree.from_uri
      path: /
      args:
        uri: mongodb://localhost:27017/metadatastore-local
        asset_registry_uri: mongodb://localhost:27017/asset-registry-local
EOF

# Start Subcontainers
docker compose -f vision/external/example-services/compose.yaml --profile dev up -d

exec /bin/bash
