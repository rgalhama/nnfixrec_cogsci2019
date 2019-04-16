__author__ = 'Raquel G. Alhama'


import os, sys, inspect
import argparse
from os.path import join, exists
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

def plot_correct_single_run(input_dir):
    '''    Plots loss per epoch.'''
    if not exists(input_dir):
        print("No file %s"%input_dir)
        return
    os.chdir(input_dir)
    fname_input = "loss.csv"
    df = pd.read_csv(fname_input, sep=";")
    grouped = df.groupby("epoch")
    agg=grouped['loss'].mean()
    std=grouped['loss'].sem()
    agg.plot(x='epoch', y='loss', yerr=std, marker="o", color="black")
    plt.grid(color='g', alpha=0.2, linestyle='-', linewidth=1)
    plt.ylim(bottom=0, top=agg.max()+1)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.savefig(join(input_dir,"loss.png"))
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="", type=str, help="Path to result files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if input_dir == "":
        actdir = os.path.dirname(__file__)
        results_dir = os.path.join(actdir, '../../../../tmp')
        name="just_testing/seed71"
        input_dir = join(join(actdir,results_dir),name)
        print("No input directory specified. Resorting to default input directory: %s."%input_dir)

    plot_correct_single_run(input_dir)
    print("Done")
