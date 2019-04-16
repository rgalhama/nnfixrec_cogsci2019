#!/usr/bin/env bash

## @author: Raquel G. Alhama
## Training script (see run_scripts_cluster for cluster nodes scripts).


##Customize to your own needs
##################################################################################################################
prepath=echo `~/Research/L2STATS/nnfixrec/`           #path to the project folder

#Params
name="deleteme_en"                                    #name of the experiment
lang="hb"                                             #language of the experiment (en or hb)
hpfile="toymodel.json"                                #config file with parameters of the model
simparamsfile="local_params.json"                       #config file with parameters of the experiment
vocsize=10                                            #if -1: use all the vocabulary; otherwise, number of words to sample from train_data_file
train_data_file=$prepath"data/clean_"$lang"_50k.txt"  #file with words and frequency counts
parallel=2                                            #number of parallel processes
outputdir="../../../tmp/"$name"/"                     #directory to output results
test_file=$prepath"/data/"$lang"_test/test_data.csv"  #words that the model will be tested on (required even if only training: just to make sure they are included in the vocabulary layer).
seed_start=1                                          #smaller seed in the range of seeds to test
seed_end=2                                            #largest seed in the range of seeds to test (included)
##################################################################################################################



# Initialize
sourcePath=$prepath"/src/"
expPath=$prepath"/src/experiments/mcconkie_nn"
cd $expPath
PYTHONPATH=$sourcePath
source activate nnfixrec



# Train
for seed in $(seq $seed_start $seed_end); do
    python3.6 train.py \
    --hyperparams_file ../../../configs/modelconfigs/$hpfile \
    --simulation_params_file ../../../configs/simconfigs/$simparamsfile \
    --additional_vocabulary_size $vocsize \
    --path_to_save_model ../../../saved_models/$name \
    --output_dir $outputdir \
    --seed $seed \
    --parallel $parallel \
    --lang $lang \
    --words_file $train_data_file \
    --test_data_file $test_file
done



source deactivate

