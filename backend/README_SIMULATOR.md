# CMS Beamline Simulator — Installation & Run Guide

This file documents how to **install and launch the CMS beamline simulator** used by the tests in `tests/test_op_cog.py`.

**Please view** the `docker-related/` directory if you want to skip doing all this setup. Specifically, check out the `docker-related/README.md`. It will do all this setup in a docker container automatically.

---

## Attribution

Parts of this simulator setup are based on work from the NSLS-II Controls group:

- **example-services** — based on [NSLS2/cms-epics-containers](https://github.com/NSLS2/cms-epics-containers)  
- **profile-collection** — based on [NSLS2/cms-profile-collection](https://github.com/NSLS2/cms-profile-collection)

We gratefully acknowledge their contributions, which provided the foundation for this simulation environment.

---

## Supported Platforms

- Linux (native)
- macOS (Intel & Apple Silicon)
- Windows (via WSL2, Ubuntu recommended)

---

## Prerequisites

| Dependency       | Purpose                                           | Install (Linux) / (macOS)                 |
|------------------|--------------------------------------------------|-------------------------------------------|
| Git              | Clone the repository & submodules                | `sudo apt install git` / `brew install git` |
| Conda            | Manage scientific Python environments            | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) |
| Docker & Compose | Run IOC & services as containers                 | [Docker](https://docs.docker.com/get-docker/) (`docker-compose` v2 included) |
| Redis server     | Used by the startup                               | `sudo apt install redis` / `brew install redis` |
| Build tools      | For Python wheels (Linux only)                   | `sudo apt install build-essential` |
| MongoDB          | Database required by the simulator/test suite     | `sudo apt install mongodb` / `brew install mongodb-community` |

---

## Repository Layout

```
beamline-simulator/
├─ external/                    # sub-repos live here
│  ├─ profile_collection/       ← code used at CMS beamline
│  └─ example-services/         ← IOC docker-compose (branch CMS-IOCs)
├─ src/
├─ tests/
└─ README_SIMULATOR.md          ← you are here
```

---

## Clone with Submodules

The two auxiliary repositories (`profile_collection` and `example-services`)
are tracked as **Git submodules**.  They are fetched automatically if you clone
with the `--recurse-submodules` flag:

```bash
git clone --recurse-submodules https://github.com/CFN-softbio/<repository_name>.git
cd <repository_name>
```

If you already cloned the main repo earlier, initialize the submodules now:

```bash
git submodule update --init --depth 1
```

### Updating to newer commits later

```bash
# inside the main repo
git submodule update --remote --merge        # pulls latest on tracked branch
git commit -am "Update submodules"
```

After either command the directories

```
external/profile_collection/
external/example-services/
```

will contain the exact revisions required by the simulator.

## Install the 2024-3.0-py311-tiled Conda Environment
Make sure to have miniconda installed in your user directory (otherwise change the paths in the following commands).
(Don't install it with sudo, will give issues)

The simulator/test code relies on the pre-built NSLS-II **2024-3.0-py311-tiled**
environment archived at Zenodo ([record 14019710](https://zenodo.org/records/14019710)).  
Two installation routes are possible:

**Option A – use the pre-built tarball (large download, instant install)**  
```bash
mkdir -p ~/miniconda3/envs/2024-3.0-py311-tiled
cd       ~/miniconda3/envs/2024-3.0-py311-tiled
wget https://zenodo.org/records/14019710/files/2024-3.0-py311-tiled.tar.gz
tar -xvf 2024-3.0-py311-tiled.tar.gz
conda activate 2024-3.0-py311-tiled
conda-unpack                           # finalize paths inside the env (if not done already)                           # finalise paths inside the env
```
• Downloads ~2.7 GB but the environment is ready almost immediately after
unpacking.

**Option B – re-create from YAML (small download, slower install)**  
```bash
# Zenodo publishes the YAML as <name>.yml.txt – we rename while downloading
curl -L \
  https://zenodo.org/records/14019710/files/2024-3.0-py311-tiled.yml.txt \
  -o env.yml
conda env create -f env.yml          # conda will now resolve & download pkgs
```
• Downloads only ~40 kB but conda will spend several minutes solving and then
fetch ~2 GB of packages.

(Checksums and alternative Python 3.10 / 3.12 variants are listed on the
Zenodo record page.)

---

## Redis Installation & Quick-Start

Redis is used by the simulator for inter-process communication.  
Follow the steps for **your platform** and verify it is reachable.

### 1 — Install

```bash
# Linux (Debian/Ubuntu)
sudo apt update
sudo apt install redis

# macOS (Homebrew)
brew install redis
```

### 2 — Start the service

```bash
# Linux systemd
sudo systemctl enable redis-server --now     # start now & on boot

# macOS
brew services start redis                    # background launch at login
```

(WSL users: run the Linux commands inside your WSL shell.)

### 3 — Verify

```bash
redis-cli ping
# → PONG   (means the server is running)
```

Optional Docker alternative:

```bash
docker run -d --name redis -p 6379:6379 redis:7
# then:  redis-cli -h 127.0.0.1 ping  # → PONG
```

If `PONG` is not returned, revisit the install/start steps before continuing.

---

## Kafka Configuration

```bash
sudo mkdir -p /etc/bluesky
sudo tee /etc/bluesky/kafka.yml >/dev/null <<'EOF'
---
abort_run_on_kafka_exception: false
bootstrap_servers:
  - localhost:9092
runengine_producer_config:
  security.protocol: PLAINTEXT
EOF
```

*Kafka does **not** need to be running for most tests, but this file must be present.*
Getting messages like `Connect to ipv6#[::1]:9092 failed: Connection refused (after 1ms in state CONNECT)` is fine.

---

## /etc/hosts Addition

Add this line to `/etc/hosts` (with `sudo`):

```
127.0.0.1 info.cms.nsls2.bnl.gov
```

---

## Environment Variables for EPICS

Add to your `~/.bashrc` or session:

```bash
export EPICS_CA_ADDR_LIST=127.0.0.1
export EPICS_CA_AUTO_ADDR_LIST=NO
```
Reload terminal or run: `source ~/.bashrc`

---

## Start IOC/Services with Docker

```bash
cd external/example-services
source environment.sh
docker compose -f compose.yaml --profile dev up -d
# To clean up if needed:
# docker kill $(docker ps -aq)
# docker rm $(docker ps -aq)
# docker network prune
```

---

## Run the Caproto Spoof IOC
Make sure to test this before using the `test_op_cog.py` so that you can resolve all errors.

```bash
cd external/profile_collection
python iocs/spoof_beamline.py
```
*(from the repo root, make sure the environment is active)*

---

## Filesystem Tweaks: Pilatus Data and /nsls2

```bash
sudo mkdir -p /nsls2/
sudo chown -R $(id -u):$(id -g) /nsls2/
mkdir -p external/example-services/pilatus-data/data
ln -s $(pwd)/external/example-services/pilatus-data/data /nsls2/
```

If you get errors using `sam.measure` (not a breaking issue since it doesn't rely on PV changes), you may need to ensure that the following symlink exists (adjust for your user/home if needed):

```bash
ln -sv $PWD/external/example-services/pilatus-data/data /nsls2/
```
The result should be something like:
```
'/nsls2/data' -> '/your/project/path/external/example-services/pilatus-data/data'
```

You may also need to create specific subdirectories for data.
By default, with the simulator inside this project, use the project-local data path:

```bash
mkdir -p external/example-services/pilatus-data/data/cms/legacy/xf11bm/Pilatus2M/<potentially also current date in 2025/03/28/ format>
mkdir -p external/example-services/pilatus-data/data/cms/legacy/xf11bm/data/2023_1/beamline/Commissioning/saxs/raw/
```

You can use the following command to make sure your underprivileged Python session can move files to the latter directory:
`sudo chmod a+w /nsls2/data/cms/legacy/xf11bm/data/2023_1/beamline/Commissioning/saxs/raw`. Use this with caution as then anyone can edit that directory.

---

## Launch IPython Simulator Session

Make sure to test this before using the `test_op_cog.py` so that you can resolve all errors.

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 2024-3.0-py311-tiled
source external/example-services/environment.sh
cd external/profile_collection
./.ci/apply-autosettings.sh
ipython --profile-dir=. --no-term-title --simple-prompt
%run -i ./.ci/linkam-drop-in.py
# sam should now be defined already
```

---

## Run the Automated Tests

```bash
pyenv activate .venv # or your venv management tool
source /home2/nvleuten/example-services/environment.sh  # preferably put this in your ~/.bashrc
cd <repo root>
python tests/test_op_cog.py \
       --base_model claude-3.5-sonnet \
       --dataset_path tests/datasets/op_cog_dataset.json
```

---

## Troubleshooting

### Timeouts / PVs missing?
- Are your IOCs up? (`docker ps` should show containers)
- Did you launch `spoof_beamline.py`?
- Is Redis running? (`redis-cli ping`)
- Is your conda environment active (`conda list` should include bluesky, caproto, etc.)
- Did you initialize the right environments, are the environment variables correct?

### Docker not running?
- On macOS/Windows, start Docker Desktop first.

### macOS PV issues?
- You may need [XQuartz](https://www.xquartz.org/) for some EPICS IOCs.

### Windows / WSL2
- Run all commands within your Ubuntu WSL2 prompt. Filesystem mount points may differ.

If in doubt: destroy all containers and restart (`docker kill $(docker ps -aq); docker rm …; docker network prune`)

### iPython launch hanging
When running `test_op_cog.py` and not much is happening for a long time, check the logs created in the model results directory.
Specifically, the `pexpect` logs. It will show you what is happening inside the iPython session which will allow you to debug.

### Permission errors
Sometimes when doing commands like `RE(scan(cms.detector, smx, 0, 10, 11))` you might encounter a permissions error, make sure that you have write access to the directory it is complaining about.

### "name 'Sample' is not defined" error"
This error indicates that something during setup has not gone well.
Possibly `spoof_beamline.sh` has been running for too long, or you're running the `test_op_cog.py` script via a tool like `tmux` which could be causing some slowdown which makes the iPython session not work correctly.
Make sure you don't have any conda environments active while running `test_op_cog.py`, and only have the `.venv` pyenv active. Having both active at the same time will interfere.

### "pymongo.errors.ServerSelectionTimeoutError"
If you encounter the following error:
```python
File "/home2/nvleuten/miniconda3/envs/2024-3.0-py311-tiled/lib/python3.11/site-packages/pymongo/synchronous/topology.py", line 376, in _select_server
    servers = self.select_servers(
              ^^^^^^^^^^^^^^^^^^^^
  File "/home2/nvleuten/miniconda3/envs/2024-3.0-py311-tiled/lib/python3.11/site-packages/pymongo/synchronous/topology.py", line 283, in select_servers
    server_descriptions = self._select_servers_loop(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home2/nvleuten/miniconda3/envs/2024-3.0-py311-tiled/lib/python3.11/site-packages/pymongo/synchronous/topology.py", line 333, in _select_servers_loop
    raise ServerSelectionTimeoutError(
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms), Timeout: 30s, Topology Description: <TopologyDescription id: 6858c1a571d3fa33ec345eb8, topology_type: Unknown, servers: [<ServerDescription ('localhost', 27017) server_type: Unknown, rtt: None, error=AutoReconnect('localhost:27017: [Errno 111] Connection refused (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms)')>]>
```

Please restart your docker example-services docker containers and it should be resolved.

### "XF:11BMB-ES{Det:PIL2M}:TIFF1:CreateDirectory_RBV"
This can go paired with being unable to do caget, caput, etc.

What worked for us is deleting all of your docker installs:
```
# Stop all running containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm -f $(docker ps -aq)

# Remove all networks
docker network prune -f

# Remove all volumes
docker volume prune -f

# Remove all images
docker rmi -f $(docker images -q)

# Remove build cache
docker builder prune -af
```

Restart your server: `sudo reboot`

If something is blocking MongoDB:
```
sudo lsof -i :27017
sudo kill -9 <PID>
```

---

*For further help, contact the controls team, or file an issue on the repository.*

# Mocking new PVs
To mock new PVs you can add a folder (Compose project) to the `./external/example-services/services` directory.
If you want to modify an existing PV, for example add/change a motor, you can go into `./external/example-services/services/bl01t-mo-sim-01/config/ioc.yaml`.

If you want to add a new motor controller, copy the previous definitions and change the `controllerName` and `P` (start of the PV address).
Make sure that the number of axes (`numAxes`) is 1 higher than the number of motors you have filled in.

Additionally, if you add new PV's that should be mocked, you want to add these to `./external/profile_collection/iocs/spoof_beamline.py` under the `__contains__` and `__missing__` to exclude them from being spoofed by the blackhole IOC (and returning 0).
You want to avoid multiple sources broadcasting on a PV, otherwise you'd get a warning like this:
```
-:~$  caget XF:11BMB-ES{SM:1-Ax:Srot}Mtr
CA.Client.Exception...............................................
    Warning: "Identical process variable names on multiple servers"
    Context: "Channel: "XF:11BMB-ES{SM:1-Ax:Srot}Mtr", Connecting to: 127.0.0.1:5064, Ignored: <address>"
    Source File: ../cac.cpp line 1320
    Current Time: Thu Jul 03 2025 12:32:18.418137235
```