import matplotlib.pyplot as plt
from myUtils.fixation_distribution import FixationDistribution
from myConstants import COLOR_PALETTE, langstr

#override general style for cogsci
COLOR_PALETTE={"en":"darkseagreen","hb":"mediumpurple"}
wordlen=7
langs=["hb", "en"]

def plot_fixation_distributions(add_model):

    if add_model:
        fig, ax = plt.subplots(2,figsize=(5,4))
    else:
        fig, ax = plt.subplots(figsize=(5,1.5))

    for lang in langs:

        axb = ax[0] if add_model else ax
        #Behavioral
        fd = FixationDistribution(wordlen, lang).BehavioralFixations(wordlen, lang)
        axb.plot(list(range(1,wordlen+1)),fd, color=COLOR_PALETTE[lang], lw=2, label=langstr[lang], marker="p")
        axb.set_ylim(0,0.26)
        axb.legend()
        axb.yaxis.tick_right()

        #Model
        if add_model:
            fixations=FixationDistribution(wordlen, lang)
            proportions=fixations.get_probs("behavioral",1)
            perwordfix=fixations.get_expfreqs_position(proportions, 20)
            ax[1].plot(list(range(1,wordlen+1)),perwordfix, ls="-", lw=2,color=COLOR_PALETTE[lang], label="Model %s"%langstr[lang], marker="p")
            ax[1].set_ylim(0,4.5)
            ax[1].yaxis.tick_right()
            ax[1].legend()

    position=(0,1) if add_model else (0,0.5)
    plt.ylabel("Proportion of\nfixations", fontsize=12, position=position)
    #plt.subplots_adjust(left=0.01, bottom=None, right=0.8, top=None, wspace=8., hspace=None)
    plt.xlabel("Letter position",fontsize=12)
    #plt.tight_layout()
    if add_model:
        plt.savefig("fixation_distributions_humans_and_model.png", bbox_inches='tight')
    else:
        plt.savefig("fixation_distribution_humans_.png", bbox_inches='tight')


if __name__ == "__main__":
    plot_fixation_distributions(False)