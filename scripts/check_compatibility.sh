#!/usr/bin/bash

old_version_indexes="v0.16.2_hgraph v0.16.2_hnsw \
                     v0.16.1_hgraph v0.16.1_hnsw \
                     v0.16.0_hgraph v0.16.0_hnsw \
                     v0.15.0_hgraph v0.15.0_hnsw \
                     v0.15.1_hgraph v0.15.1_hnsw \
                     v0.14.0_hgraph v0.14.0_hnsw \
                     v0.14.1_hgraph v0.14.1_hnsw \
                     v0.14.2_hgraph v0.14.2_hnsw \
                     v0.14.8_hgraph v0.14.8_hnsw \
                     v0.13.4_hgraph v0.13.4_hnsw \
                     v0.13.0_hnsw \
                     v0.12.0_hnsw \
                     v0.11.14_hnsw \
                     v0.10.0_hnsw"
all_success=true

for version in ${old_version_indexes}
do
  ./build-release/tools/check_compatibility/check_compatibility ${version}
  if [ $? -ne 0 ]; then
    all_success=false
    break
  fi
done

if [ "$all_success" = true ]; then
  exit 0
else
  exit 1
fi
