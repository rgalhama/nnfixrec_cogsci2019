__author__ = 'Raquel G. Alhama'

import pandas as pd
import os
from os.path import join
import numpy as np

fn_fixations_beh="../../data/fixations/%s_FixPosSum.csv"
fn_fixations_avg="../../data/fixations/%s_fixation_average.csv"

class FixationDistribution():

    def __init__(self,  wordlen, lang):
        self.wordlen=wordlen
        self.lang=lang
        self.prop_position = None

    def get_probs(self, fixationtype, weight_u):
        if fixationtype.lower() == "behavioral" or fixationtype.lower() == "behavioural":
            return self.BehavioralFixations(self.wordlen, self.lang)
        elif fixationtype.lower() == "uniform":
            return self.UniformFixations(self.wordlen, self.lang)
        elif fixationtype.lower() == "average":
            return self.WeightedAverageFixations(self.wordlen, self.lang, weight_u)
        else:
            raise Exception("I don't know fixation type %s"%fixationtype)

    def UniformFixations(self, wordlen, _):
        self.prop_position = [1/wordlen]*wordlen
        return self.prop_position

    def BehavioralFixations(self, wordlen, lang):
        actpath = os.path.dirname(os.path.realpath(__file__))
        self.path = join(actpath, fn_fixations_beh%lang)
        df = pd.read_csv(self.path, sep=",")
        df=df[df["length"]==wordlen]
        self.prop_position = [0]*wordlen
        for i in range(1,wordlen+1):
            self.prop_position[i-1] = df.loc[df.pos==i, "%s.fix.pos.sum"%lang].item()
        total=sum(self.prop_position)
        if total < 0.99:
            for i,p in enumerate(self.prop_position):
                self.prop_position[i]=p/total
        return self.prop_position


    def AverageFixations(self, wordlen, lang):
        """
        Average BehavioralFixations distribution with uniform distribution.
        :param wordlen:
        :param lang:
        :return:
        """
        actpath = os.path.dirname(os.path.realpath(__file__))
        self.path = join(actpath, fn_fixations_avg%lang)
        df = pd.read_csv(self.path, sep=";")
        self.prop_position=[float(x) for x in list(df["avg_prob"])[:-1]]
        return self.prop_position

    def WeightedAverageFixations(self, wordlen, lang, uweight):
        bf=self.BehavioralFixations(wordlen, lang)
        uf=self.UniformFixations(wordlen, lang)
        bf_rep=np.dot(bf,20)
        uf_rep=np.dot(uf, 20)
        w_avg = (uf_rep*uweight + (1-uweight)*bf_rep)/2
        w_avg = w_avg / sum(w_avg)
        return w_avg

    @staticmethod
    def get_expfreqs_position(proportions, word_repetitions):
        """ Get vector of number of word exposures in each position."""
        return np.int_(np.round_(np.dot(proportions, word_repetitions)))

    def get_word_fixation_list(self, words, perwordfix):
        word_fixation_list=[]
        # if self.fixationtype == "uniform":
        #     for w in words:
        #         for fix in range(1,self.wordlen+1):
        #             word_fixation_list.append((w,fix))
        # else:
        for w in words:
            for fix in range(self.wordlen):
                for _ in range(perwordfix[fix]):
                    word_fixation_list.append((w,fix+1))
        return word_fixation_list
