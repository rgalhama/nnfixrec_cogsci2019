    #!/usr/bin/env bash


#Set params
seed=1
mode="train"
name="outputnoise005_behavioral"
hpfile="outputnoise005.json"
simparamsfile="behavioral.json"
lang="hb"
vocsize=-1
parallel=32
#only for test
dimming=1.0

#Fixed params
prepath="/psyhome/u7/rgalhama/nnfixrec/"

if [[ $lang == "en" ]]; then
    test_file=$prepath"/data/en_test/test_data.csv"
elif [[ $lang == "hb" ]]; then
    test_file=$prepath"/data/human_experiment/responses_humans.csv"
fi

#Run
if [[ $mode == "train" ]]; then

    nohup python3.6 "/psyhome/u7/rgalhama/nnfixrec/src/experiments/mcconkey/"$mode".py" \
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

elif [[ $mode == "test" ]]; then

    #Output directory
    outputdir=$prepath"/tmp/"$name
    if (( $(awk 'BEGIN {print ("'$dimming'" <= "'1.0'")}') )); then
        echo "Testing with dimming enabled: $dimming."
        outputdir=$outputdir"_dimming"$dimming
    fi
    if [ ! -d $outputdir ]; then
        echo "Creating directory for results: "$outputdir
        mkdir $outputdir
    fi

    saved_models_dir=$prepath"/saved_models/"$name/
    epochs=`ls $saved_models_dir | sed 's/.*ep//g' | sort | uniq`
    seeds=`ls $saved_models_dir | sed 's/.*seed//g' | sed 's/_ep.*$//g'| sort | uniq`

    for seed in $seeds; do
        if [ -f $prepath"/tmp/"$name"/"$seed"means_hangman_test.csv" ]; then
            echo "Files for seed $seed already exist!"
            echo "Delete means and stress result files, or edit this code if you want to proceed."
            echo "Skipping test for seed $seed."
        else
            for epoch in $epochs; do
                nohup python3.6 "/psyhome/u7/rgalhama/nnfixrec/src/experiments/mcconkey/"$mode".py" \
                --path_to_model $saved_models_dir"/"$name"_seed"$seed"_ep"$epoch \
                --seed $seed \
                --epoch $epoch \
                --lang $lang \
                --dimming $dimming \
                --output_dir $prepath"/results/"$name \
                --test_data $test_file \
                >> output_test_"$name"_s"$seed".o &
            done
        fi
    done

fi