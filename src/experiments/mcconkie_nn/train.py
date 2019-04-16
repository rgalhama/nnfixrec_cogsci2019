""" Simulation of behavioral experiment.
    The stimuli was selected taking into account hangman entropy in edges (initial and final positions).

    Summary of behavioral experiment:
    =================================
    Stimuli: 3 types of words (all 7-letter Hebrew words):
    - extreme 'negative': 50 words where it should be much better to fixate at the beginning compared to end
    - extreme 'positive': 50 words where it should be much better to fixate at the end compared to beginning
    - 'mid' words: 100 words not from the extremes, continuously either better at the beginning or at the end

    Procedure:
    For each subject for each word, the fixation position is manipulated:
        - either at the beginning of the word (location 2/7),
        - or at the end of the word (location 6/7).


"""

__author__ = 'Raquel G. Alhama'

import os, sys, inspect
from os.path import join
import datetime
now=datetime.datetime.now
import numpy as np
npa=np.array
import torch.distributed as dist
from torch.multiprocessing import Process, Queue
#Add source folder to the path:
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)
    sys.path.insert(0, MAIN_FOLDER)
    sys.path.insert(0, SCRIPT_FOLDER)
#Own imports
from models.model1 import *
from models.nn_utils import *
from myUtils.myPlots import *
from myUtils.simulation import Simulation
from myUtils import loaders
from myUtils.fixation_distribution import *
from myUtils.misc import extend_filename_proc
from experiments.mcconkie_nn.test import *
from experiments.analyze_human_data.process_mconkieexp_responses import get_word_types, only_extreme_entropy

#Filenames
fn_training_acc = "training_correct.csv"
fn_loss = "loss.csv"
fn_hact = "hidden_activations.np"
subfolders=("training_correct", "loss", "hidden_activations")

def output_training_stats(parallel, output_dir, hact_hist_mean, ep_loss, accuracy, online_test, means_time, stress_time):

    #SAVE ALL THE INFORMATION
    print("Saving the histogram of activations of the hidden layer...")
    fn=fn_hact if parallel <= 1 else extend_filename_proc(fn_hact, dist.get_rank())
    np.savetxt(join(output_dir, "hidden_activations", fn), npa(hact_hist_mean))

    print("Saving mean loss of each epoch...")
    fn=fn_loss if parallel <= 1 else extend_filename_proc(fn_loss, dist.get_rank())
    df = pd.DataFrame(ep_loss, columns=['epoch', 'loss'])
    df.to_csv(join(output_dir, "loss", fn), sep=";")

    print("Saving training accuracy...")
    fn=fn_training_acc if parallel <= 1 else extend_filename_proc(fn_training_acc, dist.get_rank())
    df = pd.DataFrame(accuracy, columns=['epoch', 'fixation', 'correct'])
    df.to_csv(join(output_dir, "training_correct", fn), sep=";")

    if online_test and (parallel <= 1 or dist.get_rank() == 1):
        print("Save data from test and the stress, only for one process")
        means_to_csv(means_time, output_dir)
        stress_to_csv(stress_time, output_dir)


def average_gradients(model):
    """ Averages gradients and sums the losses, for Synchronous SGD. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
    param.grad.data /= size



def batch_train_step(epoch, word_fixation_list, alls2i, model, optimizer, loss_function, predicted_prob_target, predicted_distr_target, all_losses, accuracy, hact_hist_mean, hact_hist_std, parallel):



    sum_correct = {}
    total_fixation = {}
    #store hidden layer activations
    hfreqs, hbins = [], []
    bins=np.arange(0,1,0.05)

    optimizer.zero_grad()

    for wi,(word,fixation) in enumerate(word_fixation_list):
        #Prepare sequences of indexes for letters (x) and target word (y)
        x, y = prepare_input_output_seqs(word, alls2i)
        output_activation = model.forward(x, fixation)

        #Store relevant data to check convergence:
        #predicted_prob_target[word] = get_predicted_prob_target(prediction.data, word, alls2i.w2i)
        #predicted_distr_target[word] = prediction

        # Save hidden layer activation histograms
        acts=np.histogram(model.hidden_state.data.view(-1), bins=bins)
        hfreqs.append(acts[0])
        hbins.append(acts[1])

        #Save prediction data to analyze accuracy during training
        predicted_probs = pred_to_prob(output_activation)
        is_correct = int(np.argmax(predicted_probs)==alls2i.w2i[word])
        sum_correct[fixation] = sum_correct.get(fixation, 0) + is_correct
        total_fixation[fixation] = total_fixation.get(fixation, 0) + 1

        #Compute loss, save and propagate back
        target = model.idx_seq_to_var(y)
        #todo parametrize this and clean!!!!

        loss=loss_function(output_activation, target)




        # freqloss = self.hyperparams.get("frequency_in_loss",0) #move this outside #todo continue here
        # if freqloss:
        #     call a function that weights the loss with a vector proportional to frequency
        all_losses[epoch].append(loss.data.item())
        loss.backward()

    #Save mean accuracy for this batch
    for fix, total in total_fixation.items():
        results={'epoch':epoch, 'fixation': fix, 'correct': sum_correct[fix]/total}
        accuracy.append(results)

    #Average histogram of hidden layer activations, across words
    hact_hist_mean.append(np.mean(hfreqs, axis=0))
    hact_hist_std.append(np.std(hfreqs, axis=0))

    if parallel:
        average_gradients(model)

    optimizer.step()



def train(seed, word_fixation_list, wordlen, lang, simulation_params, alls2i, model, loss_function, optimizers, convergence_criteria, online_plot, online_test, output_dir, save_model_dir, parallel=False, out_queue=None):

    #Inititalize plots
    if online_plot:
        f, axarr = plt.subplots(2)

    #Initialize accumulators and other vars
    means_time, stress_time, all_losses, ep_loss, accuracy, hact_hist_mean, hact_hist_std =[], [], [], [], [], [], []
    output_every = simulation_params["output_every"]
    converged = False
    epoch=0


    if online_test:
        test_data = pd.read_csv(args.test_data_file, sep=",")

    #Train
    print("Hello, world! I'm gonna start training.")

    while not converged:
        start = datetime.datetime.now()
        print("Epoch %i"%epoch)
        sys.stdout.flush()

        #Shuffle the data
        random.shuffle(word_fixation_list)

        #Initialize accumulators for this epoch
        predicted_prob_target = {word: -1 for word,_ in word_fixation_list}
        predicted_distr_target = {word: -1 for word,_ in word_fixation_list}
        all_losses.append([])


        #Train step
        optimizer = optimizers["early"] if epoch < 10 else optimizers["main"]
        batch_train_step(epoch, word_fixation_list, alls2i, model, optimizer, loss_function, predicted_prob_target, predicted_distr_target, all_losses, accuracy, hact_hist_mean, hact_hist_std, parallel)


        #Save loss and activations of the hidden layer
        act_loss = {'epoch': epoch, 'loss':np.mean(all_losses[epoch])}
        ep_loss.append(act_loss)


        #Check convergence
        convergence_criteria.update_state(epoch, all_losses, predicted_distr_target, predicted_prob_target)
        converged=convergence_criteria.converged()

        #Online plots
        if online_plot and epoch%output_every == 0:
            # confusion_matrix(f, axarr[0], pred_to_prob_matrix(predicted_distr_target, alls2i.w2i), True)
            plot_loss(axarr[1], all_losses, True, averaged=True)

            #Plot results hangman online
            # test_results, stress = simulate_hangman_test(model, alls2i, wordlen, lang, plot=False)
            # m,e = compute_means_correct_condition(test_results)
            # myPlots.plot_hangman_mean_correct(m, e, axarr[0], output_dir=output_dir)


        #Simulate test
        if online_test and epoch > 0 and epoch%output_every == 0:
            rank = None if parallel <=1 else dist.get_rank()
            online_test(test_data, epoch, model, alls2i, wordlen, lang, seed, simulation_params, output_dir, save_model_dir, means_time, stress_time, rank)

        #Save model
        if save_model_dir is not None and save_model_dir != "" and epoch%5 == 0 and (parallel <= 1 or dist.get_rank()==1):
            subpath=join(save_model_dir,basename(save_model_dir)+"_seed%i_ep%i"%(seed,epoch))
            if not exists(subpath):
                os.makedirs(subpath)
            model.save_model(subpath, alls2i, simulation_params, epoch, seed)

        end = datetime.datetime.now()
        print("epoch time:",end-start)
        sys.stdout.flush()
        epoch+=1

    print("Training done!")

    #Plot last predictions
    if online_plot:
        confusion_matrix(f, axarr[0], pred_to_prob_matrix(predicted_distr_target, alls2i.w2i), False)
        plot_loss(axarr[1], all_losses, False)

    #Write files with collected info about training
    output_training_stats(parallel, output_dir, hact_hist_mean, ep_loss, accuracy, online_test, means_time, stress_time)

    print("The results are in %s"%args.output_dir)
    print("The saved models are in %s"%args.path_to_save_model)

    return accuracy #deprecated



def run(seed, simulation, args, word_fixation_list, lang, wordlen, parallel, out_queue, nprocs):
    """ Runs the training. This is also the entry point when running in distributed parallel processing. """

    #Initialize seed in random (shared by all processes)
    random.seed(seed)
    np.random.seed(seed)

    #Create model (each process has one model with the same seed)
    simulation.create_model()

    #Get own data partition for this process (if running in parallel)
    if parallel > 1:
        nbatches = nprocs
        datalen = len(word_fixation_list)
        batch_sizes = compute_minibatch_sizes(nbatches, datalen, verbose=True)
        random.shuffle(word_fixation_list)
        mbatched_data = data_to_batches(batch_sizes, word_fixation_list)
        data_for_this_process = mbatched_data[dist.get_rank()]
    else:
        data_for_this_process=word_fixation_list

    #Train!
    accuracy = train(seed, data_for_this_process, lang, wordlen, simulation.params, simulation.alls2i, simulation.model, simulation.loss_function, simulation.optimizers, simulation.convergence_criteria, args.online_plot, args.online_test, args.output_dir, args.path_to_save_model, parallel=args.parallel, out_queue=out_queue)

    return accuracy


def init_processes(rank, size, fn, arguments, out_queue, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(*arguments,out_queue, size)


def init_distributed(nprocs, arguments):#seed, simulation, args, word_fixation_list, lang, wordlen, parallel):
    processes = []
    out_queue = Queue() #to collect the output
    for rank in range(nprocs):
        p = Process(target=init_processes, args=(rank, nprocs, run, arguments, out_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    #Get results from queue before joining processes:
    ## warning: this causes many problems; spitting out results to a file is better than coordinating them at the end
    # for i in range(nprocs):
    #     mb_results = out_queue.get()


def start(args):

    #Supress threads (otherwise pytorch is VERY slow)
    torch.set_num_threads(1)

    #Load test words (stimuli), if online test
    wfdf=pd.read_csv(args.test_data_file, sep=",")
    wfdf=only_extreme_entropy(wfdf)
    test_words=get_word_types(wfdf)
    wfs_test = loaders.load_freq_for_words(args.words_file, test_words)

    #The word vocabulary consists on the test words, extended with the most frequent n words from the training data that complete the vocabulary size.
    vocsize = args.total_vocabulary_size
    if vocsize is None or vocsize < 0:
        wfs = loaders.load_wordfreqs_from_file(args.words_file, wordlength=args.wordlength, firstn=None)
    else:
        wfs=loaders.complete_vocabulary(args.words_file, wfs_test, args.wordlength, vocsize)
    words = set(list(wfs.keys()) + test_words)

    #Create simulation class, which stores relevant information
    simulation = Simulation(args.seed, args.simulation_params_file, args.hyperparams_file, args.lang, args.wordlength , "", None, None, args.path_to_save_model, words_with_frequencies=wfs)

    #Create list of word-fixation tuples
    fixations=FixationDistribution(args.wordlength, args.lang)
    proportions=fixations.get_probs(simulation.params["training_fixations"], simulation.params.get("weight_uniform_if_avg",1))
    perwordfix=fixations.get_expfreqs_position(proportions, simulation.params["word_repetitions_batch"])
    word_fixation_list=fixations.get_word_fixation_list(words, perwordfix)

    #Train
    if args.parallel > 1:
        init_distributed(args.parallel, [args.seed, simulation, args, word_fixation_list, args.lang, args.wordlength, args.parallel])

    else:
        run(args.seed, simulation, args, word_fixation_list, args.lang, args.wordlength, args.parallel, None, 1)

    #Save simulation params with results
    simulation.save_simulation_params(outputdir=args.output_dir)#, additional_description="Epochs trained: %i"%last_epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="Encoding of input data. Default=utf-8;  alternative=latin1")

    #Config
    parser.add_argument("--hyperparams_file", required=True, type=str, help="File with hyperparameters in json format.")
    parser.add_argument("--simulation_params_file", required=True, type=str, help="File with simulation parameters in json format.")

    #Seed
    parser.add_argument("--seed", default=None, type=int, help="Seed for training neural network.")

    #Experiment parameters
    parser.add_argument("--lang", required=True, type=str, help="Language. Currenly, non-uniform fixation distributions are only supported for Hebrew (option: hb) and English (option: en). ")
    parser.add_argument("--wordlength", default=7, type=int, help="Word length. ")

    #Saving output
    parser.add_argument("--path_to_save_model", default="", type=str, help="Path to save models at different stages of training. The program will create a subfolder structure for each model epoch that is saved. ")
    parser.add_argument("--output_dir", required=True, type=str, help="Path where plots and other results may be stored. ")

    #Training data
    parser.add_argument("--total_vocabulary_size", default=-1, type=int, help="Number of words in vocabulary learned by the model (i.e. number of classes in the output layer). It should include the number of words in the test set. This script will take the first n words in the provided training file, such that n+size(test_set)=total_vocabulary_size.  To use all the words provided in test and training data, set this parameter to -1.")
    parser.add_argument("--words_file", required=True, type=str, help="File for additional words.")


    #Modes
    parser.add_argument("--online_plot", action='store_true', help="Show loss and confidence matrix while training.")
    parser.add_argument("--online_test", action='store_true', help="Test hangman during training.")
    parser.add_argument("--parallel", type=int, default=0, help="Run synchronous SGD on multiple CPUs. State number of processes. ")

    #Test data
    parser.add_argument("--test_data_file", required=True, type=str, help="This is the file containing the words to test. These words will be included in the training vocabulary, regardless of whether they are also in the training data. Thus, this file is required even if not testing online, to ensure that these words are part of the training vocabulary for later testing.")

    args = parser.parse_args()

    #Check if provided files exist
    for f in [args.simulation_params_file, args.hyperparams_file, args.words_file]:
        if not exists(f):
            raise Exception(f, " does not exist!")

    #Create output dir and output dir structure
    args.output_dir = join(args.output_dir, "seed"+str(args.seed))
    if not exists(args.output_dir):
        os.makedirs(args.output_dir)
    for sf in subfolders:
        sfp=join(args.output_dir,sf)
        if not exists(sfp):
            os.makedirs(sfp)

    #Create path to save models, if enabled
    if args.path_to_save_model:
        if not exists(args.path_to_save_model):
            os.makedirs(args.path_to_save_model)

    if args.encoding not in ("utf-8", "latin1"):
        raise Exception("Encoding unknown: %s"%args.encoding)

    start(args)