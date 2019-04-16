"""
    Library of functions to compute entropy of words after fixating on a certain position.
    The fixation model used is McConkie et al. 1989
    Used in Alhama et al. 2019 (cogsci proceedings); see archival folder "cogsci" for frozen files and computations.
"""

__author__ = 'Raquel G. Alhama'


import sys, os, inspect
from os.path import join, exists
import codecs
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
npa=np.array

#Add source folder to the path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from models.fixation import McConkie_Model, Reader
from myUtils.loaders import CorpusLoader

#--- filenames ----
def get_pertrial_header():
    header = ";".join(["lang", "fixation", "target", "trial", "read", "entropy"])
    header += "\n"
    return header

def get_outputfile_pertrial(lang, fixation, wordlen, control=""):
    return "%s_%s_f%i_wl%i.csv"%(lang, control,fixation,wordlen)

def get_outputfile_means(lang, fixation, wordlen, control=""):
    return "means_%s_%s_f%i_wl%i.csv"%(control,lang,fixation,wordlen)

# ---------------------


def totalfreq(wordfreqfile, sep, freqpos, encoding="utf-8"):
    freqs=[]
    nl=0
    with codecs.open(wordfreqfile, encoding=encoding) as wh:
        try:
            for line in wh:
                if nl > 0:
                    fields=line.split(sep)
                    if len(fields) == 2:
                        try:
                            freqs.append(int(fields[freqpos].strip()))
                        except ValueError:
                            print(nl, ": ", line, fields[freqpos], file=sys.stderr)
                nl+=1
        except UnicodeDecodeError:
            print(nl, ": ", line, file=sys.stderr)

    return sum(freqs)


def load_knownwords(lang, wordlen, control=""):

    if control == "types":
        loader = CorpusLoader(lang, types=True)
    else:
        loader = CorpusLoader(lang)

    knownwords = loader.load_words(wordlen)

    return knownwords


def determine_reps(knownwords, mcconkie_probs, lang, fixation, wordlength):
    reps = 20 #initial number of repetitions

    while True:
        #Get a reference with the actual number of repetitions
        words_entropy =compute_entropies(lang, fixation, wordlength, mcconkie_probs, knownwords, knownwords, reps, None)
        words_entropy=words_entropy.sort_values("target")
        ref_entropies = words_entropy.groupby("target").mean()["entropy"]#.mean()
        #See correlation with one more run
        words_entropy =compute_entropies(lang, fixation, wordlength, mcconkie_probs, knownwords, knownwords, 1, None)
        words_entropy=words_entropy.sort_values("target")
        act_entropies = words_entropy.groupby("target").mean()["entropy"]#.mean()
        r=act_entropies.corr(ref_entropies)
        pval=ttest_ind(act_entropies, ref_entropies)[1]
        print(reps, "\t", r, "\t", pval)
        if r < 0.3:
            reps += 5
        else:
            return reps

def compute_entropies(lang, fixation, wordlen, fixation_params, knownwords, targets, nreps, outputfile):
    """
    Computes entropy after fixation.
    :param lang:
    :param fixation:
    :param wordlen:
    :param fixation_params: vector with probabilities of recognizing the letter in each position.
    :param knownwords:
    :param targets:
    :param nreps:
    :param outputfile: file to write output, or None.
    :return:
    """

    #Create output file if it doesn't exist; otherwise append
    fh=None
    if outputfile is not None and outputfile != "":
        if not exists(outputfile):
            fh = open(outputfile, "w")
            fh.write(get_pertrial_header())
        else:
            fh = open(outputfile, "a")

    #Create the gazebo
    gazebo = Reader(wordlen, knownwords, fixation_params, verbose=False)

    #Read and collect entropies
    data=[]
    #t1=time.time()
    for target in targets:
        for rep in range(nreps):
            read, entropy = gazebo.read(target)
            data.append([lang, fixation, target, rep, read, entropy])
            row=[lang, str(fixation), target, str(rep) ,"".join(read),"%.3f"%entropy]
            row = ";".join(row)
            row+="\n"
            if fh:
                fh.write(row)
    #t2=time.time()
    #print("Time reading targets:",t2-t1)
    df=pd.DataFrame(data, columns="lang,fixation,target,nrep,read,entropy".split(","))
    return df


def average_across_trials(infile,outfile):

        #Read per trial entropy
        df=pd.read_csv(infile, sep=";")

        #Average per repetition
        meandf = df.groupby(df["target"], as_index=True).agg('mean')

        del meandf["trial"]
        #Output into file
        meandf.to_csv(outfile,sep=";")

        #Return the data
        return meandf

def compute_difference_entropy(fname1, fname2, lang, outputfile):
    """
        Returns difference entropy (fname1[entropy] - fname2[entropy]).
        @:param fname1: filename
        @:param fname2: filename
    """

    #Read both files
    df1 = pd.read_csv(fname1, sep=";")
    df2 = pd.read_csv(fname2, sep=";")

    #Join by target
    joined = pd.merge(df1, df2, how="inner", on="target")
    joined["difference_entropy"] = joined["entropy_x"] - joined["entropy_y"]
    joined=joined.sort_values("difference_entropy")
    joined.to_csv(outputfile, sep=";")

    return joined


def compute_entropy_pertrial(knownwords, targetwords, lang, wordlength, nreps, drop, fixations, control="",encoding="utf-8"):

    outputfiles=[]
    #Read and get entropy for all the words, in each fixation
    for fixation in fixations:
        outputfile=get_outputfile_pertrial(lang,fixation,wordlength, control)
        mcc_eccentricity_vector = McConkie_Model(wordlength, drop, fixation)
        #Compute and add to file
        compute_entropies(lang, fixation, wordlength, mcc_eccentricity_vector, knownwords,
        targetwords, nreps, outputfile)
        outputfiles.append(outputfile)
    return outputfiles


def plot_entropy_difference_langs(dfdict):
    fig, ax = plt.subplots(figsize=(8,4.25))
    #plt.hist([dfdict["en"]["difference_entropy"], dfdict["hb"]["difference_entropy"]], bins=40, normed=True, label=["English","Hebrew"], color=["darkseagreen","mediumpurple"])
    sns.set_style('whitegrid')
    sns.kdeplot(dfdict["en"]["difference_entropy"], bw=0.15,  shade=True,color="darkseagreen", label="English", lw=2.5)
    sns.kdeplot(dfdict["hb"]["difference_entropy"], bw=0.15, shade=True, color="mediumpurple", label="Hebrew", lw=2.5, ls="--")
    plt.legend(fontsize=16)
    plt.xlabel("Entropy difference between fixation locations 2 and 6.", fontsize=16)
    plt.ylabel("Normalized frequency.", fontsize=16)
    plt.xlim(-7.5,7.5)
    plt.axvline(x=0., color='black', alpha=0.5, ls="--")
    plt.savefig("entropy_diffs_f2_f6_en_hb.png", bbox_inches="tight")




def entropy_whole_process(lang, wordlength, fixations, nreps, knownwords, targetwords, mcconkie_drop, control=""):
    """
    Runs the whole process of computing entropy after fixation (i.e. computes all the repetitions, and then aggregates them).
    :param control: control condition, using types instead of tokens
    """



    #STEP 1: Compute entropy per trial
    pertrial_files=compute_entropy_pertrial(knownwords, knownwords, lang, wordlength, nreps, mcconkie_drop, fixations, control )
    #(outputs in file)
    print("Step 1 done! Entropy computed, %i repetitions."%nreps)

    #STEP 2: Compute mean entropy, across trials
    fmeans={}
    for fixation in fixations:
        infile=get_outputfile_pertrial(lang, fixation, wordlength, control)
        fmeans[fixation]=get_outputfile_means(lang, fixation, wordlength, control)
        mcconkie_probs = McConkie_Model(wordlength, mcconkie_drop, fixation)
        average_across_trials(infile,fmeans[fixation])
    print("Step 2 done! Entropy averaged accross repetitions.")


    #CONTROL STEP (correlate with uniform frequencies)
    if control == "types":
        #Merge
        dfz=pd.read_csv("%s_zerofreq_entropydiff.csv"%lang, sep=";")
        dff=pd.read_csv("%s_entropydiff.csv"%lang, sep=";")
        dfz=dfz.sort_values("target")
        dff=dff.sort_values("target")
        #correlation
        print("Correlation with uniform frequency, for %s"%lang,pearsonr(dfz['difference_entropy'], dff['difference_entropy'])) #r,p-value

        #Load frequency in dataframe
        wordfreq=pd.read_csv("/home/beltza/Research/L2STATS/nnfixrec/data/clean_%s_50k.txt"%lang, sep=" ", header=None)
        wordfreq.columns=["frequency","target"]
        merged=wordfreq.merge(dff,how="right",on="target")
        print("Correlation frequency and entropy(with frequency), for %s"%lang,pearsonr(merged['difference_entropy'], merged['frequency']))
        merged=wordfreq.merge(dfz,how="right",on="target")
        print("Correlation frequency and entropy(without frequency), for %s"%lang,pearsonr(merged['difference_entropy'], merged['frequency']))
        #Correlation for English: 0.729
        #Correlation for Hebrew: 0.772
    return fmeans


def add_frequency_to_df(df, wordfreq):

    merged=wordfreq.merge(df,how="right",on="target")
    return merged
