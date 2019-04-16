__author__ = 'Raquel G. Alhama'

from math import log

def Levenshtein(w1, w2):
    if len(w1) > len(w2):
        w1, w2 = w2, w1

    distances=range(len(w1)+1)
    for i2,c2 in enumerate(w2):
        d = [i2+1]
        for i1,c1 in enumerate(w1):
            if c1 == c2:
                d.append(distances[i1])
            else:
                d.append(1+min(distances[i1], distances[i1+1],d[-1]))
        distances = d
    return distances[-1]

def computeDistancesSurprisal(target, w2i, results):
    """
    Compute Levenshtein distance and Surprisal, from one target word to each other word.
    :param target:
    :param w2i:
    :param results:
    :return:
    """
    distances=[]
    surprisal=[]
    text=[]
    for prediction in results.data:
        if prediction.input_word == target:
            print(prediction.fixation)
            for unit,prob in enumerate(prediction.predicted_probs):
                word=w2i[unit]
                surprisal.append(-log(prob))
                dist=Levenshtein(target,word)
                distances.append(dist)
                text.append(word)
            break
    return distances, surprisal, text

if __name__ == "__main__":
    w1="plain"
    w2="slain"
    print(Levenshtein(w1,w2))
