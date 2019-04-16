import sys, inspect, os
from os.path import pardir
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt

#Add source folder to the path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

wordlen=7

#Color palette for experiment conditions
condition_palette = {
    "f2neg": "salmon",
    "f2pos": "indianred",
    "f6neg": "turquoise",
    "f6pos": "teal"
}
def read_words(fn):
    words=[]
    with open(fn,"r") as fh:
        for line in fh:
            f,w=line.split()
            if len(w) == wordlen:
                words.append(w)
    return words

def count_matches(subword, words, fixation):
    acum=0
    for word in words:
        if fixation==2 and word.startswith(subword):
            acum=acum+1
            # print("%s matches %s"%(subword,word))
        elif fixation==6 and word.endswith(subword):
            acum=acum+1
            #print("%s matches %s"%(subword,word))
    return acum


def get_matches(subword, words, fixation):
    acum=[]
    for word in words:
        if fixation==2 and word.startswith(subword):
            acum.append(word)
        elif fixation==6 and word.endswith(subword):
            acum.append(word)
    return acum

def get_subword(word, fixation, windowsize):

    if windowsize == 0:
        return word[fixation-1]
    else:
        left=max(0,fixation-windowsize-1)
        right=min(len(word), fixation+windowsize)
        return word[left:right]

def plot_results(results, lang):
    fig, ax = plt.subplots()
    x=range(0,wordlen-1)
    fix=[2,6]
    cond=["neg","pos"]
    for f in fix:
        for c in cond:
            cond_abbr="f%i%s"%(f,c[-3:])
            print(results[cond_abbr])
            plt.grid(b=True, color='black', alpha=0.1, linestyle='-', linewidth=1)
            plt.plot(x,results[cond_abbr], color=condition_palette["f"+str(f)+c], ls=["-","--"][c == "neg"], label="Fix(%i)_maxIC(%i)"%(f,[2,6][c=="pos"]), marker="x")
    plt.legend()
    fontsize=14
    plt.xlabel("Window size", fontsize=fontsize)
    plt.ylabel("Chance correct", fontsize=fontsize)
    plt.ylim(0,1.01)
    plt.xlim(0,max(x))
    language="English" if lang == "en" else "Hebrew"
    #plt.title(language, fontsize=14)
    plt.savefig("chance_correct_%s.png"%lang, bbox_inches="tight")






#--- main
username="rgalhama"
prepath="/home/"+username+"/Research/L2STATS/nnfixrec/"
lang="en"
conditions=["extreme_neg", "extreme_pos"]
fixations=[2,6]

if lang == "hb":
    datapath=prepath+"/data/human_experiment/responses_humans.csv" #hb is reversed here (left to right)
else:
    datapath=prepath+"/data/en_test/test_data.csv" #en

#Read vocabulary
df = pd.read_csv(datapath, sep=",")
allwords = read_words(prepath+"data/clean_%s_50k.txt"%lang)#hb is reversed here (left to right)

#Calculate matches and chance in the limit
results={}#cond:array_results
for condition in conditions:
    conditiondf = df[df["condition"] == condition]
    words = set(conditiondf["target"])
    #words=["airplane"] #illusration for cogsci
    for fixation in fixations:
        cond_abbr="f%i%s"%(fixation,condition[-3:])
        results[cond_abbr]=[]
        for window_size in range(0,wordlen-1):
            acum=0
            for word in words:
                subword=get_subword(word, fixation, window_size)
                matches = count_matches(subword, allwords, fixation)
                acum+=matches
                #if word == "airplane" and fixation == 2 and window_size==2: #example
                    #print(word, "fixation:", fixation,"window size:",window_size, "matches: ",matches)
                 #   print(get_matches(subword, allwords, fixation))
            #print("Prob. correct: %f  for f%i%s"%(50/acum,fixation,condition))
            results[cond_abbr].append(50/acum)



plot_results(results, lang)
