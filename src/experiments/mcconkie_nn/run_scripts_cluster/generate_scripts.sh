#!/usr/bin/env bash

for script in *_template.sh; do
	for seed in {1..2}; do
        name=`cat $script | grep name= | grep -E -o "\".+\"" | sed "s/\"//g"`
        mode=`cat $script | grep mode= | grep -E -o "\".+\"" | sed "s/\"//g"`
		cat $script | sed "s/seed=.*/seed=$seed/" > "run_"$mode"_"$name"_seed"$seed".sh"
	done
done
chmod u+x *.sh
