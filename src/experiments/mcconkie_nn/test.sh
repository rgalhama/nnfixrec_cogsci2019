#!/usr/bin/env bash

## @author: Raquel G. Alhama
## Testing script, for one CPU (see run_scripts_cluster for parallel processing scripts).

##Customize to your own needs
#########################################################
prepath=echo `~/Research/L2STATS/nnfixrec/`           #path to the project folder

name="deleteme_en"                                    #name of the experiment
lang="en"                                             #language of the experiment
dimming=0.4                                           #dim the input (1. results in no dimming)
hpfile="toymodel.json"                                #config file with parameters of the model
simparamsfile="behavioral.json"                       #config file with parameters of the experiment
vocsize=10                                            #if -1: use all the vocabulary; otherwise, number of words to sample from train_data_file

#File with test data
if [[ $lang == "en" ]]; then
    test_file=$prepath"/data/en_test/test_data.csv"
elif [[ $lang == "hb" ]]; then
    test_file=$prepath"/data/human_experiment/responses_humans.csv"
fi

#Output directory
outputdir=$prepath"/tmp/"$name

#########################################################


#Initialize
sourcePath=$prepath"/src/"
expPath=$prepath"/src/experiments/mcconkie_nn"
cd $expPath
PYTHONPATH=$sourcePath
source activate nnfixrec

#Create output directory if not existing
if (( $(awk 'BEGIN {print ("'$dimming'" <= 1.0)}') )); then
    echo "Testing with dimming enabled: $dimming."
    outputdir=$outputdir"_dimming"$dimming
fi
if [ ! -d $outputdir ]; then
    echo "Creating directory for results: "$outputdir
    mkdir $outputdir
fi

#Get seeds and epochs from saved models
saved_models_dir=$prepath"/saved_models/"$name/
epochs=`ls $saved_models_dir | sed 's/.*ep//g' | sort | uniq`
seeds=`ls $saved_models_dir | sed 's/.*seed//g' | sed 's/_ep.*$//g'| sort | uniq`


#run test
for seed in $seeds; do
    #See if results files already exist, from previous test
    if [ -f $prepath"/tmp/"$name"/"$seed"means_hangman_test.csv" ]; then
        echo "Files for seed $seed already exist!"
        echo "Delete means and stress result files, or edit this code if you want to proceed."
        echo "Skipping test for seed $seed."
    else
        for epoch in $epochs; do
            #Run test
            python3.6 test.py \
            --path_to_model $saved_models_dir"/"$name"_seed"$seed"_ep"$epoch \
            --seed $seed \
            --epoch $epoch \
            --lang $lang \
            --dimming $dimming \
            --output_dir $outputdir \
            --test_data $test_file
        done
    fi
done


source deactivate

