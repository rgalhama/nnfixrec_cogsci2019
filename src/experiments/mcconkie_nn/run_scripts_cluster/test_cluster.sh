#!/usr/bin/env bash


function run_test {
    name=$1
    lang=$2
    seed=$3
    dimming=$4
    echo "Hello there! I am going to test name=$name language=$lang dimming=$dimming"

    #Fixed params
    prepath="/psyhome/u7/rgalhama/nnfixrec/"

    if [[ $lang == "en" ]]; then
        test_file=$prepath"/data/en_test/test_data.csv"
    elif [[ $lang == "hb" ]]; then
        test_file=$prepath"/data/human_experiment/responses_humans.csv"
    fi

    #Output directory
    outputdir=$prepath"/results/"$name
    if (( $(awk 'BEGIN {print ("'$dimming'" <= 1.0)}') )); then
        echo "Testing with dimming enabled: $dimming."
        outputdir=$outputdir"_dimming"$dimming
    fi
    if [ ! -d $outputdir ]; then
        echo "Creating directory for results: "$outputdir
        mkdir $outputdir
    fi

    saved_models_dir=$prepath"/saved_models/"$name/
    epochs=`ls $saved_models_dir | sed 's/.*ep//g' | sort | uniq`
#    seeds=`ls $saved_models_dir | sed 's/.*seed//g' | sed 's/_ep.*$//g'| sort | uniq`

    # Works better without parallelization (at least within seed)
#    for seed in $seeds; do
        if [ -f $prepath"/tmp/"$name"/"$seed"means_hangman_test.csv" ]; then
            echo "Files for seed $seed already exist!"
            echo "Delete means and stress result files, or edit this code if you want to proceed."
            echo "Skipping test for seed $seed."
        else
            for epoch in $epochs; do
                echo "Testing for seed $seed and epoch $epoch..."
                python3.6 "$prepath/src/experiments/mcconkey/test.py" \
                --path_to_model $saved_models_dir"/"$name"_seed"$seed"_ep"$epoch \
                --seed $seed \
                --epoch $epoch \
                --lang $lang \
                --dimming $dimming \
                --output_dir $outputdir \
                --test_data $test_file \
#                >> "output_test_$name_.o" 2> stderr_test.o
                echo "Finished seed $seed epoch $epoch"
            done
        fi
#    done
}

#main
if [ $# -eq 4 ]; then
    run_test $1 $2 $3 $4
else
    echo "Usage: "
    echo "name=simulation_name"
    echo "lang=en"
    echo "seed=42"
    echo "./test_cluster.sh $name $lang $seed dimming_value >> output_test_$name_$lang_$seed.out "
fi
