__author__ = 'Raquel G. Alhama'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
npa=np.array
import json
from os.path import join, exists
from os import mkdir
from models.fixation import *

### Refs to useful links ################
# http://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
#
# initialization
# https://stackoverflow.com/questions/48529625/in-pytorch-how-are-layer-weights-and-biases-initialized-by-default
#
#
# Saving model:
# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
########################################

class FixRecNN(nn.Module):
    def __init__(self, simulation):
        super(FixRecNN, self).__init__()

        #Set the seed
        if simulation.seed is not None:
            torch.manual_seed(simulation.seed)
        hp = simulation.hyperparams
        self.hp_dict=hp

        #Params
        ############
        self.ch_voc_size = hp["ch_voc_size"]
        self.with_embedings = hp["with_embeddings"]
        self.ch_emb_dim = hp["ch_emb_dim"]
        self.hidden_dim = hp["hidden_dim"]
        self.output_dim = hp["word_voc_size"]
        self.noise_drop = hp["noise_drop"]
        self.output_noise = hp["output_noise"]
        self.noise_mean = hp["noise_mean"]
        self.noise_stdev = hp["noise_stdev"]
        self.noise_stdev_test = hp.get("noise_stdev_test", self.noise_stdev)
        self.limvalues=(0,1)

        #Layers
        #########

        #Char Embeddings
        if self.with_embedings:
            self.ch_emb = nn.Embedding(hp["ch_voc_size"], self.ch_emb_dim)

        #Fixation
        # (doesn't require initialization)

        #hidden
        if self.with_embedings:
            i_h_size = self.ch_emb_dim * simulation.wordlen
        else:
            i_h_size = self.ch_voc_size * simulation.wordlen
        self.hidden = nn.Linear(i_h_size, self.hidden_dim)

        #output (word recognition layer)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        #state
        self.hidden_state = None

    # def one_hot_v2(batch,depth):
    #     ones = torch.sparse.torch.eye(depth)
    #     return ones.index_select(0,batch)

    @staticmethod
    def idx_seq_to_var(idx_seq):
        ts = torch.tensor(idx_seq, dtype=torch.long)
        return autograd.Variable(ts)

    def dimming(self, x, dimming_value):
        dimmed = x*dimming_value
        return dimmed

    def letters_to_onehot(self, idx_seq):
        n_values = self.ch_voc_size
        letters = torch.eye(n_values)[idx_seq]
        return autograd.Variable(letters)


    def forward(self, input_idxs, fixation_position, test_mode=False, dimming=1.0):
        '''
        :param input: list of letter idxs (e.g. [2,17,22])
        :param fixation_position: predicted word
        :return:
        '''

        #From sparse to dense
        if self.with_embedings:
            #Convert input into variable
            input_var = self.idx_seq_to_var(input_idxs)
            # Embed
            x = self.ch_emb(input_var)
        else:
            #Create one hot vectors
            x= self.letters_to_onehot(input_idxs)

        #Apply dimming if required
        if test_mode and dimming < 1.0:
            x=self.dimming(x, dimming)

        #Apply fixation filter
        if self.noise_drop > 0:
            x = self.fixation(x, fixation_position, test_mode)

        #Concatenate representations
        concat_perceived = x.view(1,x.shape[0]*x.shape[1])

        #Hidden layer: Linear + Sigmoid
        h = self.hidden(concat_perceived)
        h = torch.sigmoid(h)
        h = self.add_clipped_noise(h)
        self.hidden_state = h

        #Output layer: log softmax over word vocabulary
        o = self.output(h)
        pred = F.log_softmax(o, dim=1) #every slice along dim will sum up to 1
        return pred

    def fixation(self, x, fixation_position, test_mode=False):

        wordlen = x.shape[0]

        if isinstance(fixation_position, str) and fixation_position.upper() == "OVP":
            eccentricity = eccentricity_vector_ovp(wordlen)
        else:
            eccentricity=eccentricity_vector(wordlen, fixation_position)


        for i,letter in enumerate(x):

            #1. Scale activation
            scaling = 1. - eccentricity[i]*self.noise_drop
            scaling = max(0, scaling)
            x[i] = x[i] * scaling
            #letter_pos = x[i].argmax()

            #2. Add noise
            stdev=self.noise_stdev_test if test_mode else self.noise_stdev
            noise = np.random.normal(loc=self.noise_mean, scale=stdev, size=np.shape(letter))
            noise_tensor = autograd.Variable(torch.from_numpy(noise).float())
            x[i] += noise_tensor

        #3. Clip the unit activation
        x = torch.clamp(x, min=self.limvalues[0], max=self.limvalues[1])
        return x


    def add_clipped_noise(self, input, limvalues=(0, 1)):
         noise = autograd.Variable(input.data.new(input.size()).uniform_(0,self.output_noise))
         return torch.clamp(input+noise, min=limvalues[0], max=limvalues[1])

    def save_model(self, path, alls2i, simulation_params_dict, epochs_trained, seed, additional_description=""):
        """
        Saves the model, the mappings, and the hyperparameters.
        :param simulation:
        :param additional_description:
        :return:
        """

        if not exists(path):
            mkdir(path)

        #Save hyperparameters
        with open(join(path, "hyper_parameters.json"), "w") as fh:
            json.dump(self.hp_dict, fh, indent=4)

        #Save other simulation parameters
        with open(join(path, "simulation_params.json"), "w") as fh:
            json.dump(simulation_params_dict, fh, indent=4)

        #Save weights in a readable format, for further analyses
        for label, weights in self.state_dict().items():
            fname=join(path, "%s_weights.json"%label)
            npw=weights.numpy()
            np.savetxt(fname, npw)


        #Save other information
        with open(join(path, "simulation_description.txt"), "w") as fh:
            fh.write("Number of epochs trained: %i\n"%epochs_trained)
            fh.write("\n")
            fh.write("Seed: %i"%seed)
            fh.write(additional_description)

        #Save model
        torch.save(self.state_dict(), join(path, "model1.model"))

        #Save mappings
        mappers=["c2i","w2i"]
        for m in mappers:
            mapper=getattr(alls2i, m)
            if mapper is not None:
                if isinstance(mapper,dict):
                    for ft, actmap in mapper.items():
                        actmap.save(join(path, m + "_" + ft))
                else:
                    mapper.save(join(path, m))

