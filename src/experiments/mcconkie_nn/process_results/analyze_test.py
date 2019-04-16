__author__ = 'Raquel G. Alhama'


import os, sys, inspect
from os.path import join
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from experiments.mcconkey_nn.test import *
from myUtils.myPlots import *
from myUtils.fixation_distribution import *


def plot_aggregated_means(alldata, output_dir=""):
    for seed, df in alldata.items():
        refdf=df
        break

    p = pd.Panel(alldata)

    #Compute the means
    avgdf_=p.mean(axis=0)
    means_col = [col for col in alldata[1] if "mean" in col]
    avgdf=avgdf_[means_col]
    #Compute standard error of the means
    errordf=p.sem(axis=0)
    errordf = errordf[means_col]
    errordf.columns = [c.replace("mean", "error") for c in means_col]
    final= pd.concat([avgdf_["epoch"],avgdf[means_col], errordf], axis=1)
    final=reorder_df_by_epoch(final)
    final.to_csv(join(output_dir,"mean_performance_across_runs.csv"))

    #Plot means per epoch
    plotTestMeansxEpoch(final, title="", outdir=output_dir)

    #Barplot for single epoch(s) (aggregated over runs)
    for idx in refdf.index:
        epoch=refdf.ix[idx, "epoch"]
        dfepoch=final[final["epoch"]==epoch]
        barplot_mean_correct_test(dfepoch, output_dir=output_dir, fname="means_%iruns_epoch%i%s" % (len(seeds), epoch, ""))#signstr


def mean_stress_accross_runs(alldata, output_dir):
    #Read data and put it together in a panel
    for seed, df in alldata.items():
        refdf=df
        break

    p = pd.Panel(alldata)

    #Compute the means
    avgdf_=p.mean(axis=0)
    means_col = [col for col in refdf if "mean" in col]
    avgdf=avgdf_[means_col]
    #Compute standard error of the means
    errordf=p.sem(axis=0)
    errordf = errordf[means_col]
    errordf.columns = [c.replace("mean", "error") for c in means_col]
    final= pd.concat([avgdf_["epoch"],avgdf[means_col], errordf], axis=1)
    final=reorder_df_by_epoch(final)
    final.to_csv(join(output_dir,"mean_stress_accross_runs.csv"))
    return final

def reorder_df_by_epoch(df):
    df = df.sort_values("epoch")
    df=df.reset_index()
    del df['index']
    return df

def main(input_path, seeds, output_path):

    means_dfs={}
    stress_dfs={}
    #Plots for individual runs
    for seed in seeds:
        data_path=join(input_path,"seed%i"%seed)
        df = pd.read_csv(join(data_path, "means_hangman_test.csv"), sep=";")
        df=reorder_df_by_epoch(df)
        means_dfs[seed]=df
        plotTestMeansxEpoch(df, outdir=data_path)

        dfstress = pd.read_csv(join(data_path, "stress_hangman_test.csv"), sep=";")
        dfstress=reorder_df_by_epoch(dfstress)
        stress_dfs[seed]=dfstress
        plotStressxEpoch(dfstress, outdir=data_path)

    #Average over multiple runs and plot
    ######################################
    mean_stress_df=mean_stress_accross_runs(stress_dfs, output_path)
    plotStressxEpoch(mean_stress_df, title="", outdir=output_path)
    plot_aggregated_means(means_dfs, output_path)
    print("Done! Results in %s"%output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Path to input files.")
    parser.add_argument("--output_dir", default="", type=str, help="Path to result files. If not added, the input dir will be used.")
    parser.add_argument("--minseed", required=True, type=int, help="Lower seed number.")
    parser.add_argument("--maxseed", required=True, type=int, help="Higher seed number.")
    args = parser.parse_args()

    outputdir= args.output_dir if args.output_dir != "" else args.input_dir
    seeds=list(range(args.minseed,args.maxseed+1))
    main(args.input_dir, seeds, outputdir)
