__author__ = 'Raquel G. Alhama'


import os, sys, inspect
from os.path import join
import argparse
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

def plot_correct_single_run(input_dir,  only_beginning=False):
    '''    Plots number of correct guesses during training, for one run.    '''
    if not os.path.exists(input_dir):
        print("No file:", input_dir)
        return
    os.chdir(input_dir)
    fname_input = "training_correct.csv"
    df = pd.read_csv(fname_input, sep=";")
    df = df.loc[:,'epoch':'correct']
    if only_beginning:
        df = df[df['fixation'] <= 4]
    grouped = df.groupby("epoch")
    agg=grouped['correct'].mean()
    std=grouped['correct'].sem()
    agg.plot(x='epoch', y='correct', yerr=std, title='Mean correct during training.', marker="o", color="black")
    plt.grid(b=True, alpha=0.2, color='b', linestyle='-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean correct')
    plt.ylim(0,1.1)
    plt.savefig(join(input_dir, "mean_correct_training%s.png"%("_only_beginning" if only_beginning else "")))
    plt.cla()

def plot_correct_many_runs():
    raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="", type=str, help="Path to result files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if input_dir == "":
        actdir = os.path.dirname(__file__)
        results_dir = os.path.join(actdir, '../../../../results')
        name="adam/seed10"
        input_dir = join(join(actdir,results_dir),name)
        print("No input directory specified. Resorting to default input directory: %s."%input_dir)

    plot_correct_single_run(input_dir, only_beginning=True)
