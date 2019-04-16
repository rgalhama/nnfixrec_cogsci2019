"""
    Methods to treat responses from behavioral experiment.
"""
__author__ = 'Raquel G. Alhama'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from myUtils.myPlots import conditions_mcconkey_legend


condition_palette = {
    "f2neg": "salmon",
    "f2pos": "indianred",
    "f6neg": "turquoise",
    "f6pos": "teal"
}

def select_n_subjects(df, n):
    """ Returns the first n subjects from df.
        :param df Dataframe with responses
        :param n Number of subjects to select
        :return: dataframe with selected subjects.
    """
    #get ordered number of subjects
    subj_ids = sorted(list(set(df["sub"])))
    selected = subj_ids[:n]
    #select first n
    df= df.loc[df["sub"].isin(selected)]
    return df

def only_extreme_entropy(df):
    """
    Returns the trials that involve words with extreme values of entropy (either positive or negative).
    :param df: dataframe with responses.
    :return: dataframe only with responses of trials with extreme values of entropy.
    """
    extremes=["extreme_neg","extreme_pos"]
    df = df.loc[df["condition"].isin(extremes)]
    return df

def select_condition(df, entropy, fixation):
    """
    Returns the trials with indicated entropy and fixation position..
    :param df: dataframe with responses.
    :return: dataframe only with selected trials.
    """
    df = df.loc[df["condition"] == entropy]
    df = df.loc[df["fix.loc"] == fixation]
    return df

def get_word_fixations(df):
    selected = df[['target', 'fix.loc']]
    word_fixations_list=[tuple(x) for x in selected.to_records(index=False)]
    return word_fixations_list

def get_word_types(df):
    words = df['target'].tolist()
    return list(set(words))


def mean_correct_condition(df):
    """
    For each subject, it computes calculated the mean in each condition, and then the standard deviaiton is across these values.
    :param df:
    :return:
    """
    df = df.groupby(['condition','fix.loc'])["correctness"].aggregate(["mean", "std"])
    # dfmean = df.groupby(['condition','fix.loc'])["correctness"].aggregate(["mean"])
    # dfmeansubs = df.groupby(['sub','condition','fix.loc'])["correctness"].aggregate(["mean"])

    df["sem"]= df["std"] /len(df)
    #df_means_sub = df.groupby(['sub','condition','fix.loc'])["correctness"].aggregate(["mean", "std"])

    return df

def plot_responses(df):

    ind = np.arange(2)  # the x locations for the groups
    width = 0.35  # the width of the b
    entropy_conds=["maxIC(2)", "maxIC(6)"]
    fig, ax = plt.subplots()

    ax.yaxis.grid(b=True, color='black', alpha=0.1, linestyle='--', linewidth=1, zorder=0)

    errf="std"

    #Barplots fixation 2
    f2_means=[df.iloc[0]["mean"], df.iloc[2]["mean"]]
    f2_error=[df.iloc[0]["sem"], df.iloc[2]["sem"]]
    rects1 = ax.bar(ind - width/2, f2_means, width, yerr=f2_error, ecolor="grey", color=(condition_palette["f2neg"], condition_palette["f2pos"]), label='Fixation location: 2', zorder=3, capsize=4)
    #Barplots fixation 6
    f6_means=[df.iloc[1]["mean"],  df.iloc[3]["mean"]]
    f6_error=[df.iloc[1]["sem"], df.iloc[3]["sem"]]
    rects2 = ax.bar(ind + width/2, f6_means, width, yerr=f6_error, ecolor="grey", color=(condition_palette["f6neg"], condition_palette["f6pos"]), label='Fixation location: 6', zorder=3, capsize=4)

    conditions_mcconkey_legend(ax)

    plt.ylim((0,1.01))
    ax.set_ylabel('Mean correct', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(entropy_conds, fontsize=14)

    #plt.show()
    plt.savefig("human_responses_coloured.png", bbox_inches="tight")

if __name__ == "__main__":
    data=pd.read_csv("../../../data/human_experiment/responses_humans.csv")
    data =mean_correct_condition(data)
    plot_responses(data)
