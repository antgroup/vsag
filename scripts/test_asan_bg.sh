#!/bin/bash

pids=()
exit_codes=()
logger_files=()
parallel_tags="[diskann] [hnsw]"
othertag=""

mkdir ./log

./build/tests/unittests -d yes ${UT_FILTER} --allow-running-no-tests 2>&1 | tee ./log/unittest.log
exit_codes+=($?)
logger_files+=("./log/unittest.log")

for tag in ${parallel_tags}
do
  othertag="~"${tag}${othertag}
  ./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${tag} 2>&1 | tee ./log/${tag}.log &
  pids+=($!)
  logname="./log/"${tag}".log"
  logger_files+=($logname)
done

./build/tests/functests -d yes ${UT_FILTER} --allow-running-no-tests ${othertag} 2>&1 | tee ./log/other.log &
pids+=($!)
logger_files+=("./log/other.log")

for pid in "${pids[@]}"
do
  wait $pid
  exit_codes+=($?)
done

index=0
all_successful=true
for code in "${exit_codes[@]}"
do
  if [ $code -ne 0 ]; then
    all_successful=false
    echo ${logger_files[${index}]} "failed"
    cat ${logger_files[${index}]}
  else
    echo ${logger_files[${index}]} "success"
  fi
  ((index+=1))
done

rm -rf ./log

if [ $all_successful = true ]; then
  exit 0
else
  exit 1
fi
