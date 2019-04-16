    #!/usr/bin/env bash


#Set params
seed=1
name="outputnoise005_behavioral"
hpfile="outputnoise005.json"
simparamsfile="behavioral.json"
lang="en"
vocsize=5535
parallel=32

#Fixed params
prepath="/psyhome/u7/rgalhama/nnfixrec/"

if [[ $lang == "en" ]]; then
    test_file=$prepath"/data/en_test/test_data.csv"
elif [[ $lang == "hb" ]]; then
    test_file=$prepath"/data/human_experiment/responses_humans.csv"
fi

#Run
nohup python3.6 "$prepath/src/experiments/mcconkey/"$mode".py" \
--hyperparams_file $prepath"/configs/modelconfigs/"$hpfile \
--simulation_params_file $prepath"/configs/simconfigs/"$simparamsfile \
--path_to_save_model $prepath"/saved_models/"$name \
--output_dir $prepath"/results/"$name \
--additional_vocabulary_size $vocsize \
--seed $seed \
--parallel $parallel \
--lang $lang \
--words_file $prepath"data/clean_"$lang"_50k.txt" \
--test_data_file $test_file \
> output_"$name"_s"$seed".o &



