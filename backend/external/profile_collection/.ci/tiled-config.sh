#!/bin/bash

set -veo pipefail

# Tiled profile config:

tiled_profiles_dir="$HOME/.config/tiled/profiles/"
beamline_acronym="cms"

mkdir -v -p ${tiled_profiles_dir}

cat << EOF > ${tiled_profiles_dir}/profiles.yml
${beamline_acronym:-local}:
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

cat ${tiled_profiles_dir}/profiles.yml
