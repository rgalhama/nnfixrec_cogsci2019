""" Simulation of behavioral experiment.
    This module tests the previously trained model.

    The stimuli was selected taking into account hangman entropy in edges (initial and final positions).

    Summary of behavioral experiment:
    =================================
    Stimuli: 3 types of words (all 7-letter Hebrew words):
    - extreme 'negative': 50 words where it should be much better to fixate at the beginning compared to end
    - extreme 'positive': 50 words where it should be much better to fixate at the end compared to beginning
    - 'mid' words: 100 words not from the extremes, continuously either better at the beginning or at the end

    Procedure:
    For each subject for each word, the fixation position is manipulated:
        - either at the beginning of the word (location 2/7),
        - or at the end of the word (location 6/7).



"""
__author__ = 'Raquel G. Alhama'

import argparse
import datetime
import re
import os, sys, inspect
from os.path import join, exists, basename, dirname
import numpy as np
npa=np.array
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import random
from scipy.stats import sem

#Add source folder to the path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

#Own imports
from models.nn_utils import *
from myUtils.results import Prediction, Results
from myUtils.simulation import Simulation
from myUtils.misc import extend_filename_proc
from myStats.anova import *

fn_test_results = "results_hangman_test.csv"

def test(word_fixation_list, model, alls2i, wordlen, lang, dimming=1.0):
    """
    Test the word_fixation_list on the given model and return the predictions.

    :param word_fixation_list:
    :param model:
    :param alls2i:
    :param wordlen:
    :param lang:
    :return: results
    """

    #Initialize object to store results
    results = Results(wordlen, lang, "HangmanExp", prefix="hangmanexp_")
    #Initialize variables to store hidden layer activation
    hfreqs, hbins = [], []
    bins=np.arange(0,1,0.05)

    #Run the model
    for i,(word,fixation) in enumerate(word_fixation_list):
        x, y = prepare_input_output_seqs(word, alls2i)
        prediction = model.forward(x,fixation, test_mode=True, dimming=dimming)
        datum = Prediction(word, alls2i.w2i[word], None, fixation, prediction.view((-1)).data, hidden_state=model.hidden_state)
        results.add_datum(datum)

    return results

def online_test(responses_data, epoch, model, alls2i, wordlen, lang, seed, simulation_params, output_dir, save_model_dir, means_time, stress_time, rank):

    #Simulate the test
    test_results, stress = test_extreme_stimuli(responses_data, model, alls2i, wordlen, lang, compute_stress=True)


    #Output per-trial test results (model performance)
    fname=join(output_dir, fn_test_results)
    fname=fname if rank is None else extend_filename_proc(fname, rank)
    results_to_csv(test_results, fname)


    #Compute means over conditions
    means_cond, errors_cond = means_over_trials_per_condition(test_results)
    #Compute significance
    results_df=pd.read_csv(fname, sep=";")
    pa,pb,paxb=anova_stepwise(results_df)
    significant=paxb<0.01

    #Plot
    # fname="hangman_mean_correct_epoch%i_%s.png"%(epoch,["","significant"][#significant])
    # myPlots.plot_hangman_mean_correct(test_results, fname=join(output_dir,# fname))

    #Save means
    means_time.append([epoch, *means_cond, *errors_cond, significant])
    #errors_cond[0], errors_cond[1], errors_cond[2], errors_cond[3], significant])


    #Save stress
    stress_time.append([epoch, stress[2]["extreme_neg"], stress[2]["extreme_pos"], stress[6]["extreme_neg"], stress[6]["extreme_pos"]])

def test_extreme_stimuli(stimuli, model, alls2i, wordlen, lang, compute_stress=True, dimming=1.0):


    #Test over each of four conditions: extreme_pos/extreme_neg x fixation_2/fixation6
    entropy_conds = ["neg", "pos"]
    fixation_conds = [2, 6]

    #Accumulators
    correct={f:{e:0 for e in entropy_conds} for f in fixation_conds}
    stress={"f2mean_neg":None,"f2mean_pos":None,"f6mean_neg":None,"f6mean_pos":None}

    for fix_cond in fixation_conds:
        for ent_cond in entropy_conds:

            #Get list of words and fixation position based on actual condition
            words = stimuli[stimuli["condition"]==("extreme_%s"%ent_cond)]["target"]
            word_fixation_list=[(w,fix_cond) for w in words]

            #Test and get results for this condition
            results = test(word_fixation_list, model, alls2i, lang, wordlen, dimming=dimming)
            correct[fix_cond][ent_cond] = results.as_dict_fixation_nmaxprob()[fix_cond]

            if compute_stress:
                stress["f%imean_%s"%(fix_cond,ent_cond)] = results.average_stress()

            #Save summary (means, stds, ...)
            #results.dump_summary()


    return correct, stress


def means_over_trials_per_condition(data):
    """
    Return mean performance (mean number of correct guesses) per condition, and standard error.
    Order: f2neg, f2pos, f6neg, f6pos (first means, then standard error)
    :param data:
    :return:
    """
    entropy_conds=["neg", "pos"]
    f2_means = [sum(m)/len(m) for m in [data[2][e] for e in entropy_conds]]
    f6_means = [sum(m)/len(m) for m in [data[6][e] for e in entropy_conds]]
    f2_error = [sem(m) for m in [data[2][e] for e in entropy_conds]]
    f6_error = [sem(m) for m in [data[6][e] for e in entropy_conds]]

    return [*f2_means, *f6_means], [*f2_error, *f6_error]


def means_to_csv(data, output_dir, append=False):
    """
    Creates file with one line per epoch, and a column for the performance in each test condition, aggregated over trials.
    :param data:
    :param output_dir:
    :return:
    """
    #Create text lines
    lines=""
    for datum in data:
        line=";".join([str(d) for d in datum])
        lines+=line
        lines+="\n"
    #Write output
    fname=join(output_dir, "means_hangman_test.csv")
    if not append or not exists(fname):
        print("Creating new file for test results: %s"%fname)
        header=";".join(["epoch", "f2mean_neg", "f2mean_pos", "f6mean_neg", "f6mean_pos", "f2error_neg", "f2error_pos", "f6error_neg", "f6error_pos", "significant"])
        header+="\n"
        with open(fname, "w") as fh:
            fh.write(header)
            fh.write(lines)
    else:
        with open(fname, "a") as fh:
            fh.write(lines)

    return fname

def stress_to_csv(data, output_dir, append=False):
    lines=""
    for datum in data:
        line=";".join([str(d) for d in datum])
        lines+=line
        lines+="\n"
    fname=join(output_dir, "stress_hangman_test.csv")
    if not append or not exists(fname):
        header=";".join(["epoch", "f2mean_neg", "f2mean_pos", "f6mean_neg", "f6mean_pos"])
        header+="\n"
        with open(fname, "w") as fh:
            fh.write(header)
            fh.write(lines)
    else:
        with open(fname, "a") as fh:
            fh.write(lines)
    return fname

def results_to_csv(results_dict, fname):
    header=";".join(["trial", "fixpos", "entropy", "correct"])
    header+="\n"
    trial=0
    with open(fname, "w") as fh:
        fh.write(header)
        for fix_cond, entropies in results_dict.items():
            for ent_cond, correct in entropies.items():
                for c in correct:
                    line=";".join([str(trial), str(fix_cond), ent_cond, str(c)])
                    line+="\n"
                    fh.write(line)
                    trial+=1
    return fname

def results_to_df(results_dict):
    trial = 0
    data=[]
    for fix_cond, entropies in results_dict.items():
        for ent_cond, correct in entropies.items():
            for c in correct:
                data.append((trial, fix_cond, ent_cond, c))
                trial+=1
    labels="trial;fixpos;entropy;correct".split(";")
    return pd.DataFrame.from_records(data, columns=labels)


def load_and_test(stimuli, lang, path_to_model, seed, epoch, output_dir, dimming=1.0):
    """ Loads and tests one model. Adds additional parameters that are specific to the simulation of the McConkey experiment."""

    #Fixed parameter of the mcconnkey behavioral experiment #todo
    wordlength=7

    #Set the seed on random and on numpy
    random.seed(seed)
    np.random.seed(seed)

    #Check if output directory is correctly set
    if output_dir[-1] == "/":
        output_dir=output_dir[:-1]
    if basename(output_dir) == "seed%i"%seed:
        pass
    elif not basename(output_dir).startswith("seed"):
        if not exists(output_dir):
            os.mkdir(output_dir)
        output_dir=join(output_dir,"seed%i"%seed)
    else:
        raise Exception("Are you sure this is the output directory you want? %s"%output_dir)

    if not exists(output_dir):
        os.mkdir(output_dir)


    #Create simulation from saved model and details
    simulation = Simulation(seed, None, None, lang, wordlength , "", None, path_to_model, None)

    #Run the test
    results, stress = test_extreme_stimuli(stimuli, simulation.model, simulation.alls2i, lang, wordlength, dimming=dimming)

    #Write out results per trial (disabled for now)
    #results_to_csv(results, join(output_dir, fn_test_results))

    #Aggregate over trials, for each condition (order: f2neg, f2pos, f6neg, f6pos)
    mean,error = means_over_trials_per_condition(results)

    #Compute significance of interaction between entropy and fixation
    results_df=results_to_df(results)
    p_entropy, p_fix, p_inter, aov_table = anova_mcconkey_test(results_df)
    significant=p_inter<0.01

    #Output means, errors, and significance of interaction for this model, appending it to the file with records for each epoch
    datum=[epoch, *mean, *error, significant]
    means_to_csv([datum], output_dir, append=True)

    #Output stress, appending it to the file with records for each epoch
    datum=[epoch, stress['f2mean_neg'], stress['f2mean_pos'], stress['f6mean_neg'], stress['f6mean_pos']]
    stress_to_csv([datum], output_dir, append=True)

    print("Completed test for seed %i epoch %i! \nResults in %s"%(seed, epoch, output_dir))

def single_run_interface():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=str, default="utf-8", help="Encoding of input data. Default=utf-8;  alternative=latin1")
    parser.add_argument("--path_to_model", required=True, type=str, help="Path to saved_models folder.")
    parser.add_argument("--seed", type=int, required=True, help="Seed. ")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch.")
    parser.add_argument("--lang", type=str, required=True, help="Language (hb or en). ")
    parser.add_argument("--dimming", type=float, default=1.0, help="Apply dimming to test input. The dimming value D must be 0 < D < 1. ")
    parser.add_argument("--test_data", required=True, type=str, help="Path to csv with test data.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path where plots and other results will be stored. ")
    args = parser.parse_args()

    if args.encoding not in ("utf-8", "latin1"):
        raise Exception("Encoding unknown: %s"%args.encoding)

    #Load behavioral data from mcconkey experiment, for words to test
    stimuli=pd.read_csv(args.test_data, sep= ",")

    load_and_test(stimuli, args.lang, args.path_to_model, args.seed, args.epoch, args.output_dir)


def batch_process(test_data_file, lang, path_to_saved_models, output_dir):

    for path_to_model in os.listdir(path_to_saved_models):
        seed, epoch = re.match(".+_seed(\d+)_ep(\d+)",path_to_model).groups()

        #Load behavioral data from mcconkey experiment, for words to test
        stimuli=pd.read_csv(test_data_file, sep= ",")

        load_and_test(stimuli, lang, join(path_to_saved_models,path_to_model), int(seed), int(epoch), output_dir)


def whole_seed_interface():

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=str, default="utf-8", help="Encoding of input data. Default=utf-8;  alternative=latin1")
    parser.add_argument("--path_to_saved_models", required=True, type=str, help="Path to saved_models folder.")
    parser.add_argument("--lang", type=str, required=True, help="Language (hb or en). ")
    parser.add_argument("--dimming", type=float, default=1.0, help="Apply dimming to test input. The dimming value D must be 0 < D < 1. ")
    parser.add_argument("--test_data", required=True, type=str, help="Path to csv with test data.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path where plots and other results will be stored. ")
    args = parser.parse_args()


    if args.encoding not in ("utf-8", "latin1"):
        raise Exception("Encoding unknown: %s"%args.encoding)

    if not exists(args.path_to_saved_models):
        raise Exception("% does not exist!"%args.path_to_saved_models)

    stimuli=pd.read_csv(args.test_data, sep= ",")
    batch_process(stimuli, args.lang, args.path_to_saved_models, args.output_dir)

#This is a hardcoded method for debugging purposes, but for some reason it is MUCH faster
#in the HPC than any other interface!
def __debugging_main():
    print("WARNING: you are running the program in debugging mode. The input parameters are ignored. Change test.py in order to avoid this.", file=sys.stderr)

    #Parameters
    lang = "en"
    name = "deleteme_en"
    dimming = 0.35
    seeds=range(1,2+1)
    epochs=range(0,20+1,5)
    prepath="/home/rgalhama/Research/L2STATS/nnfixrec/"
    #prepath="/psyhome/u7/rgalhama/nnfixrec/"
    outfolder = name
    if dimming < 1.0:
        outfolder += "_dimming%.2f"%dimming

    output_dir = join(prepath,"results/%s"%outfolder)

    if lang == "hb":
        fn_stimuli="../../../data/human_experiment/stimuli_comparison_with_wiki.csv"
    else:
        fn_stimuli="../../../data/en_test/test_data.csv"

    stimuli=pd.read_csv(fn_stimuli, sep= ",")

    for seed in seeds:
        for epoch in epochs:
            path_to_model = join(prepath,"saved_models/"+name+"/"+name+"_seed%i_ep%i"%(seed,epoch))
            load_and_test(stimuli, lang, path_to_model, seed, epoch, output_dir, dimming=dimming)

if __name__ == "__main__":
    #__debugging_main()
    #whole_seed_interface()
    single_run_interface()
