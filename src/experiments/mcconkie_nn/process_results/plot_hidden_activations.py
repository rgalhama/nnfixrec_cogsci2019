__author__ = 'Raquel G. Alhama'


import os, sys, inspect
import argparse
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

def load_data(input_dir):

    if not exists(input_dir):
        raise Exception("I can't find this directory: %s"%input_dir)

    os.chdir(input_dir)
    files=os.listdir(input_dir)
    if len(files) == 1:
        activations=np.loadtxt(files[0])
    elif len(files) > 1:
        npfiles = list(filter(lambda x: x.endswith("np"), files))
        for i,f in enumerate(npfiles):
            if i == 0:
                act=np.loadtxt(f)
            else:
                act = act + np.loadtxt(f)
        activations = act / len(npfiles)
    else:
        raise Exception("Empty directory! %s"%input_dir)

    return activations

def plot_hidden_act(activations_e):
    '''    Plots histograms of hidden layer activation, per epoch.'''


    #let op: old simulations used this bin
    #bins=np.arange(-1.5,1.5,0.1)
    #so it's more useful to visualize the in this range:
    #x[15:25], y[15:25], strx:15:25]
    nepochs=activations_e.shape[0]
    bins=np.arange(0,1,0.05)
    f, axarr = plt.subplots(nepochs, sharex=True)
#    plt.figure(num=1, figsize=(128, 56))
    f.set_size_inches(10.5, 18*(nepochs/10))
    x=range(activations_e.shape[1])

    strx=[]
    for f in bins:
        strx.append("%.2f"%f)
    for epoch in range(nepochs):
        y=activations_e[epoch]
        y=y/sum(y)
        axarr[epoch].grid(b=True, color='black', alpha=0.1, linestyle='-', linewidth=1)
        axarr[epoch].bar(x, y, facecolor='black', align='edge', width=1, label="Epoch: %i"%epoch)
        plt.xticks(x, strx)
        axarr[epoch].set_ylim(0,1.1)
        legend=axarr[epoch].legend(frameon=False, handlelength=0)
        legend.get_frame().set_facecolor('none')
    plt.xlabel('Activation', fontsize=14)
    plt.ylabel('Normalized frequency', fontsize=14, position=(0,1))
    plt.suptitle('Histogram of activation values in the hidden layer, across epochs.', fontsize=16)
    plt.savefig(join(input_dir,"hidden_act.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="", type=str, help="Path to result files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if input_dir == "":
        actdir = os.path.dirname(__file__)
        results_dir = os.path.join(actdir, '../../../../tmp')
        name="deleteme/seed1/hidden_activations"
        input_dir = join(join(actdir,results_dir),name)
        print("No input directory specified. Resorting to default input directory: %s."%input_dir)

    data=load_data(input_dir)
    plot_hidden_act(data)
    print("Done")
