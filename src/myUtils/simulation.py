__author__ = 'Raquel G. Alhama'

from myUtils.loaders import *
from models.mappings.data_to_var import *
from models.model import *
from models.convergence_criteria import *



class Simulation:
    """
    Class that stores relevant information, loads the model if required, loads hyperparams and initializes w2i and c2i indexs.
    """

    #Filenames used in configuration files of *trained* models
    fn_simulation_params = "simulation_params.json"
    fn_hyperparams = "hyper_parameters.json"

    def __init__(self, seed, simulation_params_file, hyperparams_file, lang, wordlen, fixationstype, nsamples, load_model_dir, save_model_dir, words_with_frequencies):

        #Set param values
        self.seed = seed
        self.wordlen = wordlen
        self.lang = lang
        self.fixationstype = fixationstype
        self.nsamples=nsamples
        self.words_with_frequencies = words_with_frequencies
        self.words = words_with_frequencies.keys()

        #These will be later initialized (when loading or creating a model)
        self.model=None
        self.loss_function = None
        self.optimizers = None

        #Load model and simulation parameters
        if load_model_dir:
            #subfolder = "%s_wordlen%i_%s"%(lang, wordlen, fixationstype)
            #self.path_to_load_model=join(load_model_dir, subfolder)
            self.path_to_load_model=load_model_dir
            self.load_model()

        else: #Load config to create model
            self.path_to_load_model=None
            self.hyperparams = load_hyperparameters_from_file(hyperparams_file)
            self.params = load_hyperparameters_from_file(simulation_params_file)


        if save_model_dir:
            subfolder = "%s_wordlen%i_%s"%(lang, wordlen, fixationstype)
            self.path_to_save_model=join(save_model_dir, subfolder)
        else:
            self.path_to_save_model=None


        #Convergence criteria
        criteria=[]
        options={}
        for k,v in self.params["convergence"].items():
            options[k]=v
        for criterion_type in self.params["convergence"]["convergence_criteria"]:
            criteria.append(ConvergenceCriterion(criterion_type, options))
        self. convergence_criteria = ConvergenceCriteria(criteria, self.params["convergence"]["andor"])

        #Create mappings (constant for any model)
        self.create_mappings(self.words_with_frequencies.keys())

    def create_mappings(self, words):

        #Prepare mappings string-to-int
        self.alls2i = ContainerStr2IntMaps()
        self.alls2i.c2i = get_char_mapping(words)
        self.alls2i.w2i = get_word_mapping(words)

        #Complete hyperparams with info from data and simulation_utils
        self.hyperparams['ch_voc_size'] = self.alls2i.c2i.size()
        self.hyperparams['word_voc_size'] = self.alls2i.w2i.size()

    def create_model(self):

        #Instantiate model, with all the loaded parameters
        self.model = FixRecNN(self)


        #Create loss function
        if not self.params["frequency_in_loss"]:
            self.loss_function = torch.nn.NLLLoss() #Negative Log Likelihoodhttp://pytorch.org/docs/master/nn.html
                #Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
                #(we already use log softmax in the output)
        else:
            #Weighted loss, taking word frequency into account
            # The weight vector is meant to compensante for imbalanced classes
            # The common usage is to set weights such that:
            # p_i = (#instances in class i)/(# total samples in set)
            # p_i’ = 1- p_i
            # However, we do not want to compensate for frequency, therefore we compute:
            # p_i = (#instances in class i)/(# total samples in set)
            # (which, in our case, is frequency_word_i / total_counts)

            if self.words_with_frequencies is None:
                raise Exception("I don't have any information on word frequency!")
            total_freq = sum(self.words_with_frequencies.values())
            weights=np.zeros(len(self.words_with_frequencies))
            for i in range(len(self.words_with_frequencies)):
                word=self.alls2i.w2i[i]  #word order(same as mapping)
                weights[i] = self.words_with_frequencies[word]/ total_freq

            weights=torch.tensor(weights, dtype=torch.float32)
            self.loss_function=torch.nn.NLLLoss(weights)

        #Initialize optimizers
        earlyoptimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparams['learning_rate'], weight_decay=self.hyperparams.get("weight_decay",0), momentum=self.hyperparams.get("momentum", 0))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['learning_rate'], weight_decay=self.hyperparams.get("weight_decay", 0))
        self.optimizers = {"early": earlyoptimizer, "main":optimizer}

    def load_mappings(self, path):


        #Load string to int mappers
        alls2i = ContainerStr2IntMaps()
        mappers=["c2i","w2i"]
        for m in mappers:
            setattr(alls2i, m, String2IntegerMapper.load(join(path, m)))

        self.alls2i = alls2i
        self.words =  alls2i.w2i.s2i.keys()

        #Complete hyperparams with info from data and simulation_utils
        self.hyperparams['ch_voc_size'] = self.alls2i.c2i.size()
        self.hyperparams['word_voc_size'] = self.alls2i.w2i.size()



    def load_model(self, path=""):
        #https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
        if path == "":
            path = self.path_to_load_model


        self.hyperparams = load_hyperparameters_from_file(join(self.path_to_load_model,self.fn_hyperparams))
        self.params = load_hyperparameters_from_file(join(self.path_to_load_model,self.fn_simulation_params))

        self.load_mappings(path)

        #Create base model
        self.create_model()
        #Load saved parameters
        self.model.load_state_dict(torch.load(join(path, "model1.model")))




    def save_simulation_params(self, outputdir="", additional_description=""):

        #Save hyperparameters
        with open(join(outputdir, "hyper_parameters.json"), "w") as fh:
            json.dump(self.hyperparams, fh, indent=4)

        #Save other simulation parameters
        with open(join(outputdir, "simulation_params.json"), "w") as fh:
            json.dump(self.params, fh, indent=4)

        #Save other information
        with open(join(outputdir, "simulation_description.txt"), "w") as fh:
            fh.write(additional_description)


