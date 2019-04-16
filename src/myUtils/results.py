__author__ = 'Raquel G. Alhama'

from math import log
import json
import numpy as np
npa=np.array
from scipy.stats import sem
import sys,os, inspect
from os.path import join
#Add source folder to the path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)
from myConstants import TMP
from myUtils.myPlots import plot_model_output
from models.nn_utils import average_stress

def pred_to_prob(preds):
    '''
    :param preds: Vector with log predictions
    :return: numpy vector with linear probability distribution
    '''
    probs=np.exp(npa(preds))
    return probs


def entropy(Px):
    Px_nozeros = Px[Px>0]
    return -sum(map(lambda z: z*log(z,2),Px_nozeros))


class Prediction:
    '''Class that stores input word and output predicted probabilities, target probability, and entropy.'''
    def __init__(self, input_word, word_idx, freq, fixation, predicted_probs, hidden_state=None):
        self.input_word = input_word
        self.word_idx = word_idx
        self.freq = freq
        self.fixation = fixation
        #convert if prediction is in log scale
        if sum(predicted_probs) < 0.99 or sum(predicted_probs)> 1.01:
            self.predicted_probs = pred_to_prob(predicted_probs)
        else:
            self.predicted_probs = predicted_probs
        self.target_prob = self.predicted_probs[word_idx]
        self.entropy = entropy(self.predicted_probs)

        # store hidden state and compute stress
        if hidden_state is not None:
            self.hidden_state=hidden_state.data.view((-1)).numpy()
            self.hidden_stress = average_stress(self.hidden_state)



    def to_csv_row(self, index=None):
        row =  ";".join((self.input_word, str(self.word_idx), str(self.freq), str(self.fixation), str(self.target_prob), str(self.entropy)))
        row+="\n"
        if index:
            row = str(index) + ";" + row
        return row

    def plot_prediction_probs(self, alls2i, output_path):
        '''Calls function that plots the output probabilities.'''
        plot_model_output(self.predicted_probs, self.entropy, self.input_word, alls2i, self.fixation, output_path)

class Results:
    """
        Class that stores predictions and computes summary statistics over them.
    """
    def __init__(self, wordlen, lang, fixationtype, output_path=None, prefix=""):
        """
        Class that stores the results of simulations.
        :param wordlen: word length
        :param lang: language
        :param fixationtype: uniform, behavioral, or other. Will be used for names of files.
        :param output_path: where summaries of data will be stored
        :param prefix: optional prefix to prepend to output files
        """
        self.data = []
        self.wordlen = wordlen
        self.lang = lang
        self.fixationtype = fixationtype
        if output_path:
            self.path_dump = output_path
        else:
            self.path_dump=TMP
        self.prefix=prefix

    def add_datum(self, datum):
        self.data.append(datum)

    def as_dict_fixation_entropy(self):
        m={fixation:[] for fixation in range(1,self.wordlen+1)}
        for datum in self.data:
            m[datum.fixation].append(datum.entropy)
        return m

    def as_dict_fixation_targetprob(self):
        m={k:[] for k in range(1,self.wordlen+1)}
        for datum in self.data:
            m[datum.fixation].append(datum.target_prob)
        return m

    def as_dict_fixation_nmaxprob(self):
        m={k:[] for k in range(1,self.wordlen+1)}
        for datum in self.data:
            m[datum.fixation].append(1 if np.argmax(datum.predicted_probs)==datum.word_idx else 0)
        return m


    def average_stress(self):
        s = []
        for datum in self.data:
            s.append(datum.hidden_stress)
        s=npa(s)
        return s.mean()#, sem(s)

    def entropy_position_means_stds(self):
        dm, ds = {}, {}
        m=self.as_dict_fixation_entropy()
        for fixposition,allentropies in m.items():
            dm[fixposition] = np.mean(allentropies)
            ds[fixposition] = np.std(allentropies)
        return dm, ds

    def targetprob_position_means_stds(self):
        dm, ds = {}, {}
        m=self.as_dict_fixation_targetprob()
        for i,r in m.items():
            dm[i+1] = np.mean(r)
            ds[i+1] = np.std(r)
        return dm, ds

    def n_targets_over_50(self):
        m=self.as_dict_fixation_targetprob()
        ncorrect={}
        for i,fr in m.items():
            ncorrect[i+1] = sum([1 for a in fr if a>=0.5])
        return ncorrect

    def n_targets_maxprob(self):
        m=self.as_dict_fixation_nmaxprob()
        ncorrect={}
        for i,fr in m.items():
            ncorrect[i+1] = sum(m[i])
        return ncorrect

    def dump_raw_data(self):
        with open(self.path_dump+'raw_data_wordlen%i_%s_%s.csv'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
            header =  ";".join(("", "input_word", "word_idx", "frequency", "fixation_position", "target_probability", "entropy"))
            header+= "\n"
            fh.write(header)
            i=1
            for datum in self.data:
                row=datum.to_csv_row(index=i)
                fh.write(row)
                i+=1

    def dump_summary(self):

        #Entropy summary
        dm,ds = self.entropy_position_means_stds()
        with open(self.path_dump+self.prefix+'entropy_means_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
            json.dump(dm, fh)
        with open(self.path_dump+self.prefix+'entropy_stdev_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
            json.dump(ds, fh)

        #Target probability summary
        # dm,ds = self.targetprob_position_means_stds()
        # with open(TMP+'targetprob_means_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
        #     json.dump(dm, fh)
        # with open(TMP+'targetprob_stdev_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
        #     json.dump(ds, fh)

        #Number of targets with prob over 50%
        ncorrect=self.n_targets_over_50()
        with open(self.path_dump+self.prefix+'ncorrect_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
            json.dump(ncorrect,fh)


        #Number of targets that get max prob
        nmaxarg=self.n_targets_maxprob()
        with open(self.path_dump+self.prefix+'nmaxprob_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'w') as fh:
            json.dump(nmaxarg,fh)


    def load_summary(self):
        """ Loads stored results from a model. Useful for plotting. """
        with open(self.path_dump+self.prefix+'entropy_means_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
            dm=json.load(fh)

        with open(self.path_dump+self.prefix+'entropy_stdev_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
            ds=json.load(fh)

        # with open(self.path_dump+'targetprob_means_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
        #     dmt=json.load(fh)
        # with open(TMP+'targetprob_stdev_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
        #     dst=json.load(fh)

        with open(self.path_dump+self.prefix+'ncorrect_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
            ncorrect=json.load(fh)

        with open(self.path_dump+self.prefix+'nmaxprob_worlden%i_%s_%s.txt'%(self.wordlen, self.fixationtype, self.lang), 'r') as fh:
            nmaxprob=json.load(fh)
        return dm, ds, ncorrect, nmaxprob



