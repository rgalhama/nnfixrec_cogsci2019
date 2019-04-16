#!/usr/bin/env bash

for seed in {2..6}; do

qsub -l hostname=neurocomp"$seed" cluster_script_seed"$seed".sh

#sleep 1m

done;
