#!/bin/bash

pids=()
exit_codes=()
parallel_tags="[diskann] [hnsw]"
othertag=""

./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests --shard-count 1 --shard-index 0
pids+=($!)

for tag in ${parallel_tags}
do
  othertag="~"${tag}${othertag}
  ./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${tag} &
  pids+=($!)
done

./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${othertag} &
pids+=($!)

for pid in "${pids[@]}"
do
  wait $pid
  exit_codes+=($?)
done

all_successful=true
for code in "${exit_codes[@]}"
do
  if [ $code -ne 0 ]; then
    all_successful=false
    break
  fi
done

if [ $all_successful = true ]; then
  exit 0
else
  exit 1
fi
