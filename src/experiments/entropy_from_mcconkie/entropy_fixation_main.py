__author__ = 'Raquel G. Alhama'


from experiments.entropy_from_mcconkie.entropy_fixation_lib import *


if __name__ == "__main__":

    #############################################################################################
    ## PARAMS
    #############################################################################################
    seed=1
    wordlength=7
    drop=0.25
    fixations=[2,6]
    lang="hb"
    control="" #options: "", "types"
    nreps=20 #result of step 2
    freq_file="/home/rgalhama/Research/L2STATS/nnfixrec/data/freqxmillion_clean_%s_50k.txt"%lang
    #############################################################################################

    random.seed(seed)

    #1. Load vocabulary
    knownwords = load_knownwords(lang, wordlength)

    #2. Determine the number of repetitions
    skip_step=False
    if not skip_step:
        #test in english, fixation 2,
        fixation = 2
        mcconkie_probs = McConkie_Model(wordlength, drop, fixation)
        #Warning: this process is slow (much slower than the implementation used to output the final computations)
        reps = determine_reps(knownwords, mcconkie_probs, lang, fixation, wordlength)
        #RESULTS:
        # for hb, fixation 2: 10 reps 	 0.275650118413
        # 20 	 0.31914477094
        # 60 	 0.341830254533 	 0.619075772673
        # for hb, fixation 6: 10 	 0.502537048865
        # for en, fixation 6: 15 	 0.550516545018
        # for en, fixation 2: 15 	 0.290199427349
        # 20     0.312927348629
        # So we settle on 20 repetitions.


    #2. Compute entropy at different fixation positions and average across reps
    targetwords=knownwords
    skip_step=False
    if not skip_step:
        print("Computing entropy...")
        fmeans=entropy_whole_process(lang, wordlength, fixations, nreps, knownwords, targetwords, drop, control=control)
        print("Entropy computation done." )

    #3. Compute difference between fixations (fix(2) = fix(6))
    skip_step=False
    if not skip_step:
        print("Computing difference between positions 2 and 6...")
        outputfile="%s_entropydiff.csv"%lang
        if control=="types":
            outputfile="%s_zerofreq_entropydiff.csv"%lang
        fix2=get_outputfile_means(lang, 2, wordlength, control)
        fix6=get_outputfile_means(lang, 6, wordlength, control)
        compute_difference_entropy(fix2, fix6, lang, outputfile)
        print("Done. Find the output at %s"%outputfile)

    #4. Add frequency
    skip_step=False
    if not skip_step:
        print("Adding frequency information into the file...")
        wordfreq=pd.read_csv(freq_file, sep=" ", header=None)
        wordfreq.columns=["frequency","target"]
        df_entropy=pd.read_csv("%s_entropydiff.csv"%lang, sep=";")
        df_freq=add_frequency_to_df(df_entropy, wordfreq)
        df_freq.to_csv("%s_entropydiff_withfreq.csv"%lang)
        print("Frequency added.")

    #5 Plot differences, for multiple languages
    skip_step=True
    langs_plot=["en","hb"]
    if not skip_step:
        dfdict={}
        for lang in langs_plot:
            dfdict[lang]=pd.read_csv("%s_entropydiff.csv"%lang, ";")
        plot_entropy_difference_langs(dfdict)
