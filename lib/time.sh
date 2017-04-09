#!/bin/bash

make
rm -r results/
mkdir results/
for procs in {1..12}
do
  echo "Testing $procs processors"
  for testNo in {1..5}
  do
    echo "Test #$testNo for $procs processors"
    { time ./k2.out -d data.csv -p 12 -t 16 ; } 2> "results/procs-$procs-$testNo.out"
  done
done
