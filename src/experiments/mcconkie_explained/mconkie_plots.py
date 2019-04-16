import sys, os, inspect
from os.path import join
import matplotlib.pyplot as plt

#Add source folder to the path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from models.fixation import McConkie_Model

fixations = [2,6]
wordlen=7
FONTSIZE=13

def label_bars(axis, rects):
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        axis.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def lineplot(with_step):

    fig, axarr = plt.subplots(1,2, figsize=(8,4))
    axarr[0].set_title("Fixation at position %i"%fixations[0], fontsize=FONTSIZE)
    axarr[1].set_title("Fixation at position %i"%fixations[1], fontsize=FONTSIZE)


    axarr[0].set_ylabel("Probability of recognizing letter", fontsize=FONTSIZE, position=(0,0.5))
    x=range(1,wordlen+1)
    for i,f in enumerate(fixations):

        #step mcconkey
        if with_step:
            y = McConkie_Model(wordlen,0.25,f)
            for j,lp in enumerate(y):
                if lp > 0:
                    y[j] = 1
            step = axarr[i].plot(x, y, color='#c73d32', label="step", marker="s")
            axarr[i].set_xticks([1,2,3,4,5,6,7])
            label_bars(axarr[i],axarr[i].patches)
            axarr[i].set_ylim(0,1.1)

        #Original mcconkey
        y = McConkie_Model(wordlen,0.1,f)
        original = axarr[i].plot(x,y, color='#DBA520', label="drop=0.1", marker="p")
        label_bars(axarr[i],axarr[i].patches)

        #McConkie with drop=0.25
        y = McConkie_Model(wordlen,0.25,f)
        ours = axarr[i].plot(x,y, color='#7F4618', label="drop=0.25", marker="o")
        label_bars(axarr[i],axarr[i].patches)
        axarr[i].set_ylim(0,1.1)
        axarr[i].set_xticks([1,2,3,4,5,6,7])
    plt.legend()


    #fig.tight_layout()
    axarr[1].set_xlabel("Letter position" , fontsize=FONTSIZE, position=(0,0))
    #plt.show()
    plt.savefig("mconkie_explained.png", bbox_inches="tight")


def plot_2_panels(with_step):

    fig, axarr = plt.subplots(1,2, figsize=(8,4))
    axarr[0].set_title("Fixation at position %i"%fixations[0], fontsize=FONTSIZE)
    axarr[1].set_title("Fixation at position %i"%fixations[1], fontsize=FONTSIZE)


    axarr[0].set_ylabel("Probability of\nrecognizing letter", fontsize=FONTSIZE, position=(0,0.5))
    x=range(1,wordlen+1)
    for i,f in enumerate(fixations):

        if with_step:
            #step mcconkey
            y = McConkie_Model(wordlen,0.25,f)
            for j,lp in enumerate(y):
                if lp > 0:
                    y[j] = 1
            step = axarr[i].bar(x, y, color='#c73d32')
            axarr[i].set_xticks([1,2,3,4,5,6,7])
            # axarr[row,i].text(wordlen-f-0.25,0.8,'Fixation position: %i'%f, fontsize=13, bbox=dict(facecolor='none', edgecolor='black'))
            label_bars(axarr[i],axarr[i].patches)
            axarr[i].set_ylim(0,1.1)

        #Original mcconkey
        y = McConkie_Model(wordlen,0.1,f)
        original = axarr[i].bar(x,y, color='#DBA520', alpha=0.9)
        label_bars(axarr[i],axarr[i].patches)

        #McConkie with drop=0.25
        y = McConkie_Model(wordlen,0.25,f)
        ours = axarr[i].bar(x,y, color='#7F4618', alpha=.9)
        label_bars(axarr[i],axarr[i].patches)
        #axarr[row,i].text(wordlen-f-0.25,0.8,'Fixation position: %i'%f, fontsize=13, bbox=dict(facecolor='none', edgecolor='black'))
        axarr[i].set_ylim(0,1.1)
        axarr[i].set_xticks([1,2,3,4,5,6,7])

    if with_step:
        plt.legend((original, ours, step),('drop=0.1','drop=0.25','step'), ncol=1, loc=2, bbox_to_anchor=(1.05, 1), mode="expand",  borderaxespad=0)
    else:
        plt.legend((original, ours),('drop=0.1','drop=0.25'), ncol=1, loc=2, bbox_to_anchor=(1.05, 1), mode="expand",  borderaxespad=0)


    #fig.tight_layout()
    axarr[1].set_xlabel("Letter position" , fontsize=FONTSIZE, position=(0,0))
    plt.show()
    #plt.savefig("mconkie_explained.png", bbox_inches="tight")

def plot_4_panels():
    #PLOT
    fig, axarr = plt.subplots(2,len(fixations), figsize=(8,8))
    axarr[0,0].set_title("Fixation at position %i"%fixations[0], fontsize=FONTSIZE)
    axarr[0,1].set_title("Fixation at position %i"%fixations[1], fontsize=FONTSIZE)
    ax_mc=axarr[0,1].twinx()
    ax_mc.set_yticks([])
    ax_mc.set_ylabel("McConkie", fontsize=FONTSIZE)

    ax_smc=axarr[1,1].twinx()
    ax_smc.set_yticks([])
    ax_smc.set_ylabel("Step McConkie", fontsize=FONTSIZE)

    axarr[0,0].set_ylabel("Probability of recognizing letter", fontsize=FONTSIZE, position=(0,0))
    x=range(1,wordlen+1)
    row=0
    for i,f in enumerate(fixations):
        #Original mcconkey (overlay)
        y = McConkie_Model(wordlen,0.1,f)
        original = axarr[row,i].bar(x,y, color='#DBA520')
        label_bars(axarr[row,i],axarr[row,i].patches)
        #McConkie with drop=0.25
        y = McConkie_Model(wordlen,0.25,f)
        ours = axarr[row,i].bar(x,y, color='#7F4618')
        label_bars(axarr[row,i],axarr[row,i].patches)
        #axarr[row,i].text(wordlen-f-0.25,0.8,'Fixation position: %i'%f, fontsize=13, bbox=dict(facecolor='none', edgecolor='black'))
        axarr[row,i].set_ylim(0,1.1)
        axarr[row,i].set_xticks([1,2,3,4,5,6,7])
        axarr[row,i].legend((original, ours),('drop=0.1','drop=0.25'))

    #step mcconkey
    row=1

    for i,f in enumerate(fixations):
        y = McConkie_Model(wordlen,0.25,f)
        for j,lp in enumerate(y):
            if lp > 0:
                y[j] = 1
        axarr[row,i].bar(x,y, color='#DBA520')
        axarr[row,i].set_xticks([1,2,3,4,5,6,7])
        # axarr[row,i].text(wordlen-f-0.25,0.8,'Fixation position: %i'%f, fontsize=13, bbox=dict(facecolor='none', edgecolor='black'))
        label_bars(axarr[row,i],axarr[row,i].patches)
        axarr[row,i].set_ylim(0,1.1)

    #fig.tight_layout()
    axarr[1,1].set_xlabel("Letter position" , fontsize=FONTSIZE, position=(0,0))
    #plt.show()
    plt.savefig("mconkie_explained.png", bbox_inches="tight")


#main
lineplot(False)