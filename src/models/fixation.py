'''
    Fixation model described in:
    McConkie, G. W., Kerr, P. W., Reddix, M. D., Zola, D., & Jacobs,A. M.   (1989).   Eye movement control during reading:  II. frequency of refixating a word.Perception & Psychophysics,46(3),245–253

'''

__author__ = 'Raquel G. Alhama'

import random
import math
import re
import subprocess

def eccentricity_vector_ovp(wordlen):
    ''' Eccentricity assuming that the fixation is the Optimal Viewing Position
        (i.e. it lands in the center letter for odd word length and one to the left
        for even word lengths.
    '''
    if wordlen % 2 == 0:
        middle=(wordlen//2)-1
    else:
        middle=wordlen//2
    ecc=[0.]*wordlen
    for i in range(wordlen):
        ecc[i]=middle-i
    for i in range(middle,wordlen):
        ecc[i]=i-middle
    return ecc

def eccentricity_vector(wordlen, fixation_position):
    '''

    :param wordlen:
    :param fixation_position: human-readable fixatin position (i.e. from 1)
    :return:
    '''
    assert(fixation_position>=0 and fixation_position<=wordlen)
    ecc=[None]*wordlen
    ecc[fixation_position-1] = 0
    #left to fixation:
    for i in range(fixation_position-1):
        ecc[i]=fixation_position-1-i
    #right to fixation
    for i,pos in enumerate(range(fixation_position,wordlen)):
        ecc[pos]=(i+1)
    return ecc

def McConkie_Model(wordlen, dropoff, fixation_position="OVP"):
    '''
    Model of decay based on McConkie (1989).
    When a word is of even length, then we assume that the
    center is a bit to the left (due to Optimal Viewing Position)
    :param wordlen: int, length of the word
    :param dropoff: float 0<=dropoff<=1, amount deducted for every letter position away from the center
    :param fixation_position position where eye fixates
    :return:
    '''
    params=[1.]*wordlen
    if isinstance(fixation_position, str) and fixation_position.upper() == "OVP":
        eccentricity = eccentricity_vector_ovp(wordlen)
    else:
        eccentricity=eccentricity_vector(wordlen, fixation_position)
    probs = [max(1.-dropoff*ecc,0.) for ecc in eccentricity]
    return(probs)

class Reader:

    def __init__(self, length, knownwords, params, verbose=False):
        '''
        Creates a Gazentropy instance.
        Each Gazentropy model reads only words of a certain length.


        :param length:  (int) length of the words that this model reads
        :param knownwords: (dict: word:freq, with len(word) == length) words known by the model. They should be of the length specified in the parameter.
        :param params: (list of float, len(params) == length) probability of reading the letter of each position in the word.
        '''

        self.length = length

        for word in knownwords.keys():
            assert(len(word) == length)
        self.knownwords=knownwords

        for param in params:
            assert(0<= param and param <=1)
        self.params=params

        self.verbose = verbose

        if verbose:
            print("Gazentropy created, for word length %i"%self.length)
            print("Params:",self.params)
            print("Number of known words: %i\n"%len(self.knownwords))


    def read(self, target):

        if len(target) != self.length:
            raise Exception("%s: I can't read a word of length %i!"%(target, len(target)))

        if self.verbose:
            print("I'm gonna attempt to read target word '%s'."%(target))

        #Read, based on probabilities
        read = ["_"]*self.length
        read_idxs = []
        for i,letter in enumerate(target):
            if random.random() < self.params[i]:
                read[i] = target[i]
                read_idxs.append(i)
        if self.verbose:
            print("Read:", read)

        #Get words that match the letters we managed to read
        matches = self.get_matches(target, read_idxs)
        if self.verbose:
            print("Matches:", matches)

        #Compute entropy based on matches
        entropy = self.entropy_for_matches(matches)
        if self.verbose:
            print("Entropy:", entropy)

        return read, entropy

    # def get_matches(self, target, read_idxs):
    #     #Warning: this is a slow method (but there's not much you can do around it...
    #     matches = []
    #     for word in self.knownwords.keys():
    #         match = True
    #         for idx in read_idxs:
    #             if word[idx] != target[idx]:
    #                 match = False
    #                 break
    #         if match:
    #             matches.append(word)
    #     return matches

    def get_matches(self, target, read_idxs):
        lregexp=["."]*len(target)
        for idx in read_idxs:
            lregexp[idx]=target[idx]
        strregexp="".join(lregexp)
        regexp=re.compile(strregexp)
        matches = filter(regexp.match, self.knownwords.keys())
        return list(matches)

    def get_matches_from_file(self, target, read_idxs, filename):
        lregexp=["."]*len(target)
        for idx in read_idxs:
            lregexp[idx]=target[idx]
        strregexp="".join(lregexp)
        # regexp=re.compile(strregexp)
        # matches = filter(regexp.match, self.knownwords.keys())
        command ="grep -c "+strregexp+" "+filename
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output

    def entropy_for_matches(self, matches):
        freqs = [float(self.knownwords[w]) for w in matches]
        total = sum(freqs)
        probs = [f/total for f in freqs]
        entropy= -sum(map(lambda z: z*math.log(z,2),probs))
        return entropy


if __name__ == "__main__":
    from src.utils.loader import load_wordfreqs_from_dir
    length=6
    wordfreqs_dir = "/home/rgalhama/Research/L2STATS/l2stats/data/token_freq_open_subtitles/raw"
    lang="en"
    knownwords = load_wordfreqs_from_dir(wordfreqs_dir, lang, wordlength=length, rev_hb=False)
    params=McConkie_Model(7,0.25,fixation_position=4)
    gazebo = Reader(length, knownwords, params, verbose=True)
    read, entropy = gazebo.read("phones")
    print(read, entropy)
    print(params)
