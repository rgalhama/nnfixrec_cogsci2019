__author__ = "Raquel G. Alhama"

"""
= Original: =
files: en_50k.txt , hb_50k.txt

= Cleaning: =
files: clean_en_full.txt, clean_en_50k.txt, ...

Note: the order of the columns is frequency-word in the 50k files, and word-frequency in the full list. I don't recall if I exchanged the order myself or those were the originals. The following commands were adapted accordingly.

Note2: We set the locale to C language, so that the expression [a-z] is for ascii only (and thus it doesn't grep accented letters)

* HB: 
I removed the words in latin alphabet (over 72K in the full corpus!)

LANG=C  grep  -E -v '^[a-z]+\s[0-9]+$' he_full.txt > aux
mv aux clean_hb_full.txt
Number of word types removed: 72085
Remaining number of word types in the clean file: 1095536 
Total before cleaning: 1167621 

Same was done in the 50K file (but reversing the columns).


* ENGLISH:
For english, I removed any non-latin alphabet characters (most notably, accented letters and ñ):

 > aux 
mv aux clean_en_full.txt
Total number of word types removed: 51971
Remaining number of word types in the clean file: 903242
Total number of word types before cleaning: 955213

For 50K list (reversed columns):
LANG=C  grep  -E '^[0-9]+\s[a-z]+$' clean_en_50k.txt > aux
mv aux clean_en_50k.txt
After this, we are left with 49644 words (8145 words of length 7).

= Types =
files: clean_en_50k_types.txt, ...
All words in the clean_50k corpora with frequency >=1 are converted to 1.
This was just a trick to run computations over types.

= Normalized Frequency =
files: freqxmillion_en_50k.txt
Frequency normalized to words per million. When deriving the normalization term, we use the total of the full list, not of the top 50k.

"""

from os.path import join
import pandas as pd


data_folder = "../../data/"
data_folder_external = "/home/rgalhama/Data_Research/opensubtitles_hermitdave/"

def compute_freq_x_million(lang):

    fname = "clean_%s_50k.txt"
    fname_full = "clean_%s_50k.txt"
    outfname = "freqxmillion_clean_%s_50k.txt"

    # Load data (topmost and full)
    data = pd.read_csv(join(data_folder, fname%lang), header=None, names=["freq","word"], sep=" ")
    fullfname=join(data_folder_external, "clean_%s_full.txt"%lang)
    fulldata = pd.read_csv(fullfname, header=None,names=["word","freq"],sep=" ")


    # Convert frequencies to xMillion
    total=sum(fulldata["freq"])
    normalization_factor = 1000000/total
    data["FreqxMillion"] = data["freq"]*normalization_factor


    # Export
    data=data[["FreqxMillion","word"]]
    data.to_csv(join(data_folder, outfname%lang), sep=" ", header=False, index=False)

    print("Done. The output is in ",join(data_folder, outfname%lang))


#Main
compute_freq_x_million("hb")
compute_freq_x_million("es")
