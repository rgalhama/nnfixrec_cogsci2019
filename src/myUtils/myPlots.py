"""
    Module with plotting functions.
"""


__author__ = 'Raquel G. Alhama'

from os.path import join, exists
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
npa=np.array
from numpy.polynomial.polynomial import polyfit
from myConstants import TMP, COLOR_PALETTE
from scipy.stats import pearsonr, sem

#Color palette for experiment conditions
condition_palette = {
    "f2neg": "salmon",
    "f2pos": "indianred",
    "f6neg": "turquoise",
    "f6pos": "teal"
}
COLOR_PALETTE={"hb": "#7570b3", "en":"#1b9e77", "es": "#d95f02"}
FONTSIZE=14

def jitter(arr, am=.0075):
    stdev = am*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plot_target_probs(predicted_probs_target, loss, ordered_words, online):
    """Plots the probability assigned to the target unit, for the whole batch of targets.
        It also shows the current loss.
        Useful to run while training, to inspect model.

        :param online: it pauses the image in order to be overrun. Useful for training.
    """
    plt.cla()
    data=[predicted_probs_target[w] for w in ordered_words]
    plt.bar(ordered_words, data)
    plt.ylim(0,1.2)
    plt.xticks(rotation=90)
    #plt.scatter(x.data.numpy(), y.data.numpy())
    #plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #plt.text(0.5, 1.1, 'Loss=%.4f' % loss.item(), fontdict={'size': 18, 'color':  'green'})
    if online:
        plt.pause(0.1)
    else:
        plt.savefig(TMP+"predicted_target_probs.png")

def confusion_matrix(fig, ax1, predmat, online):
    """
    Plots a heatmap of the activation in the output layer, for a batch of targets.
    Targets are shown in order (so correct activation corresponds to diagonal)
    :param fig:
    :param ax1:
    :param predmat:
    :param online: it pauses the image in order to be overrun. Useful for training.
    :return:
    """
    ax1.cla()
    im = ax1.imshow(predmat, clim=(0,1))#, linewidths=.5)
    #Colorbars have their own axes!
    cax = fig.add_axes([0.7, 0.53, 0.05, 0.35])
    plt.colorbar(im, cax=cax, orientation='vertical')#todo this is deprecated and ugly but works for now

    if online:
        plt.pause(0.1)
    else:
        plt.savefig(TMP+"final_training_pred.png")

def plot_loss(ax2, all_losses, online, averaged=True):
    """
    Plots the loss of the model.
    :param ax2:
    :param all_losses:
    :param online:
    :param averaged:
    :return:
    """
    ax2.cla()
    all_losses=npa(all_losses)
    if averaged:
        ax2.plot(np.arange(all_losses.shape[0]),np.mean(all_losses, axis=1))#, marker="o")
    else:
        all_losses=npa(all_losses).flatten()
        plt.plot(np.arange(all_losses.shape[0]),all_losses)
    if online:
        plt.pause(0.1)
    else:
        plt.show()
        #plt.savefig(TMP+"loss.png")

def general_barplot_with_error(means, stds):
    """
    Barplot, meant for fixation position and entropy. x: fixation position. y: mean (and stdev)
    means: dictionary, with x labels as keys
    stds: dictionary, with x labels as keys
    """
    fig,ax = plt.subplots()
    plt.errorbar(means.keys(), means.values(), yerr=stds.values(), linewidth=2, elinewidth=0.4)
    plt.show()


def plot_model_output(probabilities_output, entropy, word, alls2i, fixation_position, output_path):
    fig,ax=plt.subplots(figsize=(30,20))
    fontsize=40
    #plt.bar(np.arange(0,probabilities_output.size),probabilities_output)
    #plt.ylim(0,1.)
    x_labels=[alls2i.w2i[i] for i in range(probabilities_output.size)]
    x=range(probabilities_output.size)
    plt.plot(x,probabilities_output, lw=3)# color=COLOR_PALETTE["en"])
    plt.xticks(x, x_labels, rotation='vertical', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title("Word: \"%s\"      Fixation position: letter %i    Entropy: %.2f"%(word,fixation_position, entropy), fontsize=fontsize)
    plt.ylabel("Probability",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_path+"/"+"%s.pdf"%word)

def load_and_plot_n_maxprob_target(results, lang, ax):
    """
    Plots the number of target units that had the maximum probability, in the given axes.

    :param results:
    :param lang:
    :param ax:
    :return:
    """
    entropy_means, entropy_stds, ncorrect, nargmax = results.load_summary()
    ax.plot(range(1,len(nargmax)+1), nargmax.values(), label=lang, color=COLOR_PALETTE[lang])

def load_and_plot_ncorrect(results, lang, ax):
    """
    Plots the number of target units that had activation over 50\%, in the given axes.

    :param results:
    :param lang:
    :param ax:
    :return:
    """
    entropy_means, entropy_stds, ncorrect, nargmax = results.load_summary()
    ax.plot(range(1,len(ncorrect)+1), ncorrect.values(), label=lang, color=COLOR_PALETTE[lang])

def load_and_plot_entropy(results, lang, ax):
    """
    Plots entropy per fixation position, in the given axes.
    :param results:
    :param lang:
    :param ax:
    :return:
    """
    entropy_means, entropy_stds, ncorrect, nargmax = results.load_summary()
    ax.errorbar(range(1,len(entropy_means)+1),entropy_means.values(), yerr=entropy_stds.values(), linewidth=2, elinewidth=0.4, label=lang, color=COLOR_PALETTE[lang])

def plot_stat_over_fixation_position(plot_type, results, wordlen, fixationtype, ylim=None):
    """
    Plots some measure per fixation position.
    If fixationtype==both, then it contrasts two types of fixation distributions (e.g. uniform vs. based on behavioral proportions).
    :param plot_type: n_correct, n_maxprob_target, entropy
    :param results:
    :param wordlen:
    :param fixationtype:
    :param ylim:
    :return:
    """
    if plot_type.lower() == "n_correct":
        plotfunc = load_and_plot_ncorrect
    elif plot_type.lower() == "n_maxprob_target":
        plotfunc = load_and_plot_n_maxprob_target
    elif plot_type.lower() == "entropy":
        plotfunc = load_and_plot_entropy
    else:
        raise Exception("Unknown type of plot: %s"%plot_type)

    if fixationtype.lower() == "both":
        fig, axes = plt.subplots(1,2, sharey=True)
        for fn,fix in enumerate(("uniform", "behavioral")):
            for lang in ("en", "hb"):
                plotfunc(results, lang, axes[fn])
                axes[fn].set_title(fix)
    else:
        fig,ax = plt.subplots()
        for lang in ("en", "hb"):
            plotfunc(results, lang, ax)

    if ylim is not None:
        plt.ylim((ylim))
    plt.legend()
    plt.savefig(TMP+"fixation_%s_worlden%i_%s.png"%(plot_type, wordlen, fixationtype))

def barplot_mean_correct_test(data, created_ax=None, output_dir="", fname="barplot_means.png"):
    """
    Creates barplot of model performance for the 4 conditions (fixations 2 and 6, conditions neg and pos).
    :param
    :return:
    """
    entropy_conds=["maxIC(2)", "maxIC(6)"]
    fixations, conditions = [2,6], ["neg","pos"]
    fixmeansdict={2:[], 6:[]}
    fixerrdict={2:[], 6:[]}
    meancol = "f%imean_%s"
    errorcol = "f%ierror_%s"
    for f in fixations:
        for c in conditions:
            fixmeansdict[f].append(data[meancol%(f,c)].values[0])
            fixerrdict[f].append(data[errorcol%(f,c)].values[0])

    #plot
    ind = np.arange(len(fixations))  # the x locations for the groups
    width = 0.35  # the width of the b

    if not created_ax:
        fig, ax = plt.subplots(figsize=(3.5,3.5))
    else:
        ax=created_ax
#    ax.yaxis.grid(b=True, color='black', alpha=0.1, linestyle='--', linewidth=1, zorder=0)
    #Barplots fixation 2
    rects1 = ax.bar(ind - width/2, fixmeansdict[2], width, yerr=fixerrdict[2], ecolor="grey",color=(condition_palette["f2neg"],  condition_palette["f2pos"]), label='Fixation location: 2', capsize=4)#, zorder=3)
    #Barplots fixation 6
    rects2 = ax.bar(ind + width/2, fixmeansdict[6], width, yerr=fixerrdict[6], ecolor="grey", color=(condition_palette["f6neg"], condition_palette["f6pos"]), label='Fixation location: 6', capsize=4)#, zorder=3)
    plt.ylim((0,1.01))

    #Adapt this by hand for paper tile
    #ax.set_ylabel('Mean correct', fontsize=13)
    ax.set_xticks(ind)
    ax.set_xticklabels([])
    #ax.set_xticklabels(entropy_conds, fontsize=13)
#    ax.set_title("Uniform", fontsize=13)
    #conditions_mcconkey_legend(ax, location="inside")


    if not created_ax:
        fname=join(output_dir, fname)
        plt.savefig(fname, bbox_inches="tight")
        print("Saved plot in %s"%fname)

def conditions_mcconkey_legend(ax, location="outside"):

    m1, = ax.plot([], [], c=condition_palette["f2neg"] , marker='s', markersize=20,
                  fillstyle='left', linestyle='none')

    m2, = ax.plot([], [], c=condition_palette["f2pos"] , marker='s', markersize=20,
                  fillstyle='right', linestyle='none')

    #---- Define Second Legend Entry ----

    m3, = ax.plot([], [], c=condition_palette["f6neg"] , marker='s', markersize=20,
                  fillstyle='left', linestyle='none')

    m4, = ax.plot([], [], c=condition_palette["f6pos"] , marker='s', markersize=20,
                  fillstyle='right', linestyle='none')

    if location == "outside":
        plt.legend(((m1,m2),(m3,m4)), ["Fixation position 2", "Fixation position 6"], numpoints=1, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=13)
    else:
        plt.legend(((m1,m2),(m3,m4)), ["Fixation position 2", "Fixation position 6"], loc="upper right", ncol=1, fontsize=13)


def plotTestMeansxEpoch(data, title="", outdir=""):
    """
    Plots Means of McConkey Experiment, along time (epochs).
    Epochs are in x axis; correct guesses are in y.
    :param data: dataframe loaded from csv with epoch, means, error and significance
    It should be ordered by epoch!
    :param title: title for the plot
    :param outdir: output dir to store the plot
    :return:
    """
    fig, ax = plt.subplots()
    x=data["epoch"]
    fix=[2,6]
    cond=["neg","pos"]
    for f in fix:
        for c in cond:
            meancol = "f%imean_%s"%(f,c)
            errorcol = "f%ierror_%s"%(f,c)
            plt.grid(b=True, color='black', alpha=0.1, linestyle='-', linewidth=1)
            plt.errorbar(jitter(x),data[meancol],yerr=data[errorcol], color=condition_palette["f"+str(f)+c], ls=["-","--"][c == "neg"], label="Fix(%i)_maxIC(%i)"%(f,[2,6][c=="pos"]))
    plt.legend()
    fontsize=14
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Mean Correct", fontsize=fontsize)
    plt.ylim(0,1.01)
    plt.xlim(0,max(x))
    plt.title(title, fontsize=fontsize)
    fname = join(outdir, "means_epoch.png")
    plt.savefig(fname)
    print("Saved plot in %s"%fname)
    plt.clf()

def plotStressxEpoch(data, title="", outdir=""):
    """
    Plots mean stress of hidden layer when training, along time (epochs).
    :param data: pandas dataframe, ordered by epoch
    :param title: title for the plot
    :param outdir: output dir to store the plot
    """

    fig, ax = plt.subplots()
    x=data["epoch"]
    fix=[2,6]
    cond=["neg","pos"]
    for f in fix:
        for c in cond:
            meancol = "f%imean_%s"%(f,c)
            plt.grid(b=True, color='black', alpha=0.1, linestyle='-', linewidth=1)
            plt.errorbar(jitter(x),data[meancol],yerr=sem(data[meancol]), color=condition_palette["f"+str(f)+c], ls=["-","--"][c == "neg"])
    plt.legend()
    fontsize=14
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Mean Correct", fontsize=fontsize)
    plt.xlim(0,max(x))
    plt.ylim(0,1.01)
    plt.title(title, fontsize=fontsize)
    fname = join(outdir, "stress_epoch.png")
    plt.savefig(fname)
    print("Saved plot in %s"%fname)
    plt.clf()

def plotLevensteinSurprisal(target, distances, surprisals, texts, fname="levenshtein_surprisal.pdf"):
    """
    Plots the probability of words given the Levenstein distance to the target
    """
    fig,ax=plt.subplots(figsize=(30,20))
    fontsize=50
    plt.scatter(distances, surprisals, color='grey', s=[fontsize*10]*len(distances))
    annotated=[]
    for i, txt in enumerate(texts):
        if distances[i] not in annotated:
            ax.annotate(txt, (distances[i], surprisals[i]), fontsize=fontsize)
            annotated.append(distances[i])
    #add regression line
    b, m = polyfit(distances, surprisals, 1)
    y=b+m*npa(distances)
    plt.plot(distances, y, '-')
    r = pearsonr(distances, surprisals)[0]
    plt.title(r"$\rho$ = %.4f"%r, fontsize=fontsize)
    plt.xlabel("Levenshtein distance to target word: %s"%target, fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.ylabel("Surprisal", fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.savefig(fname)
    plt.clf()
