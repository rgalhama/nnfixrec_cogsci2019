#!/usr/bin/env bash

source activate nnfixrec
source_path=`pwd`

#Adapt to own needs
expname="en_uniform20_dimming0.35"
user='rgalhama'
tosharepath="/home/rgalhama/Escritorio/to_share/"$expname
path=`echo ~/Research/L2STATS/nnfixrec/results/${expname}`
#path to copy only plots
######################

seeds=`ls -d $path/seed*| sed 's/.*seed//g' | sort | uniq`
minseed=`echo -e $seeds | tr "\ " "\n"| sort -r | tail -1`
maxseed=`echo -e $seeds | tr "\ " "\n"| sort -r | head -1`
#or: for seed in $(eval echo "{$minseed..$maxseed}"); do

if [[ ! -d $path ]]; then
    mkdir $path
fi

cd $path
echo `pwd`

###########################################
#Download data from cluster, if necessary #
###########################################

#scp -r rgalhama@neurocomp.utsc.utoronto.ca:/psyhome/u7/rgalhama/nnfixrec/results/$expname $path

#####################
# Analyze results   #
#####################
function joinfiles {
        cd $1
        echo "Joining files ""$1"
        head -n 1 $1_proc0.csv > $1.csv
        awk FNR!=1 $1_proc* >> $1.csv
        cd ..
}

#Aggregate results and plot the accuracy during training (fixating at positions 1 to 4) and the loss
#for seed in $seeds; do
#    if [[ -d 'seed'"$seed" ]]; then
#        echo 'Processing files for seed '"$seed"
#        cd 'seed'$seed
#        joinfiles "training_correct"
#        joinfiles "loss"
#        python $source_path/plot_accuracy_training.py --input_dir `pwd`"/training_correct"
#        python $source_path/plot_loss.py --input_dir `pwd`"/loss"
#        python $source_path/plot_hidden_activations.py --input_dir `pwd`"/hidden_activations"
#        cd ..
#    fi
#done

#Plot the performance per condition, at certain timesteps, and in the whole training process
cd $source_path
python analyze_test.py --input_dir $path --minseed $minseed --maxseed $maxseed


##################
#Copy only plots #
##################

if [[ ! -d $tosharepath ]]; then
    mkdir -p $tosharepath
fi
#rsync -av --progress $path/ $tosharepath --exclude *.csv --exclude *.txt --exclude *.np
