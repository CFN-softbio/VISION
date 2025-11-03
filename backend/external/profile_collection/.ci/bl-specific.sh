#!/bin/bash

set -veo pipefail

# Per https://stackoverflow.com/a/66982842:
sudo echo "127.0.0.1 info.cms.nsls2.bnl.gov" | sudo tee -a /etc/hosts
cat /etc/hosts

ping -c 5 info.cms.nsls2.bnl.gov
echo "" | telnet info.cms.nsls2.bnl.gov 6379

# Copy config files into the dummy IPython profile:
mkdir -v -p ~/.ipython/profile_test/startup/
cp -v startup/.cms_config ~/.ipython/profile_test/startup/

# Create pilatus and /nsls2 dir trees:
sudo mkdir -v -p /nsls2/
sudo chown -R -v $USER: /nsls2/
mkdir -v -p $HOME/cms-epics-containers/pilatus-data/data/
ln -sv $HOME/cms-epics-containers/pilatus-data/data/ /nsls2/
