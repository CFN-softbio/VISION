#!/bin/bash

set -veo pipefail

# Kafka config:
cat << EOF > kafka.yml
---
abort_run_on_kafka_exception: false
bootstrap_servers:
    - localhost:9092
runengine_producer_config:
    security.protocol: PLAINTEXT
EOF

mkdir -v -p $HOME/.config/bluesky/
mv -v kafka.yml $HOME/.config/bluesky/kafka.yml
cat $HOME/.config/bluesky/kafka.yml
