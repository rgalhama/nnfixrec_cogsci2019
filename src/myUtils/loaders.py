__author__ = 'Raquel G. Alhama'

import json
import os, inspect, sys, codecs
from os.path import join


def load_hyperparameters_from_file(fname):
    with open(fname, "r") as fh:
        hyper_params = json.load(fh)
    return hyper_params

class CorpusLoader:


    #Default filenames
    fname_regexp_normalized = "freqxmillion_clean_%s_50k.txt"
    fname_regexp_nonnormalized = "clean_%s_50k.txt"
    fname_regxp_types = "clean_%s_50k_types.txt"

    def __init__(self, lang, normalized=True, types=False):
        self.lang=lang



        #Set path to file to load, using defaults and provided parameters.
        #For overriding, call set_filepath.
        self.datapath = self.__get_datapath()
        if normalized:
            self.fname = self.fname_regexp_normalized%lang
        else:
            self.fname = self.fname_regexp_nonnormalized%lang
        if types:
            if normalized:
                raise NotImplementedError
            else:
                self.fname = self.fname_regxp_types%lang

        self.fpath=join(self.datapath, self.fname)
        self.normalized = normalized
        self.types = types

    def __get_datapath(self):

        SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
        MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir)
        datapath=join(MAIN_FOLDER,"data")
        return datapath

    def set_filepath(self, filepath):
        """ Use to override defaults."""
        self.fpath=filepath

    def load_words(self, wordlength, minfreq=0, maxfreq=None, vocsize=-1, encoding="utf-8"):
        """
            Loads words from preprocessed list of words from Open Subtitle corpora
            (which are ordered in descending frequency, and in the format "frequency word\n",
            without header).
            :param wordlength: word length to load; None to load all the words
            ...
        """
        knownwords = {}
        with codecs.open(self.fpath, encoding=encoding) as kf:
            for line in kf:
                fields = line.split()
                if len(fields) == 2:
                    fr, w = fields[0].strip(),fields[1].strip()
                    fr = int(fr) if not self.normalized else float(fr)
                    if maxfreq is not None and fr > maxfreq:
                        continue
                    if not wordlength or len(w) == wordlength:
                        knownwords[w] = fr
                    if fr < minfreq or (vocsize > 0 and len(knownwords.keys()) >= vocsize):
                        break
        return knownwords



#TODO move to class (requires lots of change in code though)
def load_wordfreqs_from_file(wordfreqs_file, wordlength=None, minfreq=0, encoding="utf-8", rev=False, firstn=None):
    allw=[]
    with codecs.open(wordfreqs_file,"r", encoding) as f:
        for line in f:
            wf = line.strip().split(" ")
            if len(wf) < 2:
                print("Warning: odd line %s"%line, file=sys.stderr)
            else:
                if wf[0].isdigit():
                    f,w = float(wf[0]), wf[1]
                else:
                    w,f = wf[0], float(wf[1])
                if rev:
                    w=w[::-1]
                if (wordlength == None or wordlength == len(w)) and f>=minfreq:
                    allw.append((w,f))
                    if firstn is not None and len(allw) >= firstn:
                        return dict(allw)
    return dict(allw)


def complete_vocabulary(filename, initial_words, wordlength, vocsize, encoding="utf-8"):
    """
    Reads words of length <wordlength>, from <filename>, until the number_of_words_read+len(initial_words)=vocsize. Returns a dictionary of words and their frequency.

    :param filename: text file, without header, with "frequency word\n" format
    :param initial_words: dictionary with initial words:freq that have to be in the final selection
    :param wordlength: int, length of words we consider
    :param vocsize: vocabulary size to achieve
    :return: dictionary of words:frequency
    """
    wfs=initial_words.copy()

    with codecs.open(filename, encoding=encoding) as kf:
        nl = 0
        for line in kf:
            fields = line.split()
            if len(fields) != 2:
                raise Exception("File %s is not well formatted. I expected frequency word format."%filename)
            fr, w = fields[0].strip(),fields[1].strip()
            fr = int(fr)
            if len(w) == wordlength:
                if w in wfs.keys():
                    continue
                else:
                    wfs[w]=fr
                    if len(wfs) == vocsize:
                        return wfs
    return wfs

def load_freq_for_words(filename, word_list, encoding="utf-8"):

    wf = {}
    with codecs.open(filename, encoding=encoding) as kf:
        for line in kf:
                fields = line.split()
                if len(fields) == 2:
                    fr, w = fields[0].strip(),fields[1].strip()
                    if w in word_list:
                        wf[w]=int(fr)
                    if len(wf) == len(word_list):
                        return wf
    return None
