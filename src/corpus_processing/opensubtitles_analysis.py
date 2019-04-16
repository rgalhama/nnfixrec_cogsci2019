from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from myUtils.myPlots import COLOR_PALETTE

data_folder = "../../data/"
#Visualize
langs = ["en", "hb"]

def getlogfreq(freqcol):
    newcol = freqcol + 1
    newcol = newcol.apply("log")
    return newcol

def density_wordcounts(langs, wordlength, log):
    addendum=""
    for lang in langs:
        fname ="freqxmillion_clean_%s_50k.txt"
        data = pd.read_csv(join(data_folder, fname%lang), header=None, names=["freq","word"], sep=" ")
        if type(wordlength) is int and wordlength > 0:
            mask = (data['word'].str.len() == wordlength)
            data=data.loc[mask]
            addendum="\nWord length = "+str(wordlength)
        data["logfreq"] = getlogfreq(data["freq"])
        column='logfreq' if log else 'freq'
        #Density plot (can be higher than 1 because it's the area under the curve what sums up to 1)
        sns.distplot(tuple(data[column]), color=COLOR_PALETTE[lang], hist=True, label=lang)

    plt.legend()
    plt.xlabel("log(freq+1)" if log else 'Word counts')
    if not log:
        plt.xlim(0,5)
    plt.title("Word counts in Open Subtitles \n(for most frequent 50K word types).%s"%addendum)
    plt.savefig("log_freqs_opensubtitle_50K_wordlen-%s.pdf"%(str(wordlength)))
    plt.show()


#---main---
log=True
for wl in range(1,11):
    density_wordcounts(langs, wl, log )

density_wordcounts(langs, None, log)