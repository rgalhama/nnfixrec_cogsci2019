""" Helpful functions for neural network simulations. """
__author__ = 'Raquel G. Alhama'

import numpy as np
npa=np.array
from math import log, floor


def pred_to_prob(preds):
    '''
    :param preds: Output tensor from logsoftmax
    :return: numpy vector with linear probability distribution
    '''
    probs=np.exp(npa(preds.data)).reshape((-1))
    return probs

def pred_to_prob_matrix(prediction_targets, w2i):
    """ Returns a matrix (list of lists) with the probability of each target (index). Useful for plotting.
    :param prediction_targets:
    :param w2i:
    :return:
    """
    predmat = []
    for idx in range(w2i.size()):
        predmat.append(pred_to_prob(prediction_targets[w2i[idx]]))
    return predmat

def prepare_idx_sequence(seq, mapper):
    idxs = [mapper[w] for w in seq]
    return idxs


def prepare_input_output_seqs(word, alls2i):
    #Input representation
    charbased_word_var = prepare_idx_sequence(list(word), alls2i.c2i)
    #Output representation
    word_var = prepare_idx_sequence([word], alls2i.w2i)
    return charbased_word_var, word_var

def get_predicted_prob_target(preds, target, w2i):
    '''
    Returns the probability of the target word.
    :param preds: torch Variable
    :param target: string
    :return:
    '''
    probs=np.exp(npa(preds.data)).reshape((-1))
    return probs[w2i[target]]

def online_train_step(epoch, input_samples, alls2i, model, optimizer, loss_function, \
                      predicted_prob_target, predicted_distr_target, all_losses ):
    for wi,word in enumerate(input_samples):
        x, y = prepare_input_output_seqs(word, alls2i)
        prediction = model.forward(x)
        predicted_prob_target[word] = get_predicted_prob_target(prediction.data, word, alls2i.w2i)
        predicted_distr_target[word] = prediction
        #prediction_matrix.append(prediction.data.view(-1).numpy())
        loss=loss_function(prediction, model.idx_seq_to_var(y))

        all_losses[epoch].append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def average_stress(layer):
    """Computes the average stress of a layer """
    s = np.zeros(len(layer))
    for i,unit in enumerate(layer):
        if unit == 0:
            unit = 0.0000000001
        if unit == 1:
            unit = 0.999999999
        s[i] = stress(unit)
    return s.mean()

def stress(a):
    """
    Computes stress of a unit.
    :param a: activation of a unit
    """
    stress_a = a * log(a, 2) + (1-a) * log(1-a, 2) - log(0.5, 2)
    return stress_a

def compute_minibatch_sizes(nbatches, datalen, verbose=False):
    resto = datalen % nbatches
    min_batch_size = floor(datalen / nbatches)
    batch_sizes = [min_batch_size]*nbatches
    for i in range(resto):
        batch_sizes[i]+=1
    if verbose:
        print("Full batch size: %i Number of mini-batches: %i"%(datalen,nbatches))
        print("Batch_sizes: ", batch_sizes)
    assert(sum(batch_sizes) == datalen)
    return batch_sizes

def data_to_batches(batchsizes, data):
    mbatched_data = []
    ant=0
    for bsz in batchsizes:
        act=ant+bsz
        mbatched_data.append(data[ant:act])
        ant=act
    assert(batchsizes == [len(b) for b in mbatched_data])
    return mbatched_data


