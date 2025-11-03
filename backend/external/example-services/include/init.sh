#!/bin/bash

root=$(realpath $(dirname $0)/..)

echo "setting up configuration file for pva gateway"

cat $root/services/pvagw/config/pvagw.template |
  sed -e "s/172.20.255.255/$CA_BROADCAST/g" \
      -e "s/5075/$EPICS_PVA_SERVER_PORT/g" > \
      $root/services/pvagw/config/pvagw.config

echo "setting up configuration file for phoebus"

cat $root/services/phoebus/config/settings.template |
  sed -e "s/5064/$EPICS_CA_SERVER_PORT/g" \
      -e "s/5065/$EPICS_CA_REPEATER_PORT/g" \
      -e "s/5075/$EPICS_PVA_SERVER_PORT/g" > \
      $root/services/phoebus/config/settings.ini