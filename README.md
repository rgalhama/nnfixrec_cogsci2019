# Entropy in Visual Word Recognition

Code used in the paper:

Raquel G. Alhama, Noam Siegelman, Ram Frost &amp; Blair C. Armstrong (2019). The Role of Entropy in Visual Word Recognition: A Perceptually-Constrained Connectionist Account. _Proceedings of the 41st Annual Conference of the Cognitive Science Society_. [CogSci2019]

Includes: 
- Study 1: information-theoretic analyses of corpora, incorporating occulomotor constraints 
- Study 2: a perceptually-constrained neural network model of word recognition 

Implemented by [Raquel G. Alhama](https://rgalhama.github.io/) (rgalhama@mpi.nl).


## Prerequisites

* Python3.6
* Pytorch (and other requirements specified in environment.yml)
* Patience :) 

## Getting started
The quick recipe:

* Install Miniconda:

https://conda.io/docs/user-guide/install/index.html

* Create an environment using the provided environment.yml file: 

```bash
conda env create python=3.6 -f environment.yml 
source activate nnfixrec
```

Alternatively, you can install the required libraries as specified in requirements.txt:
```bash
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

* You may need to install PyTorch separately:
```bash
conda install --name nnfixrec pytorch-cpu torchvision-cpu -c pytorch
```


## Usage

For the neural network model:
1. Specify the model hyperparameters in a json file. See examples in configs/modelconfigs
1. Specify params of the training in another json file. See examples in configs/simconfigs
1. Train, test, and analyze results. Good entry points are train.sh, test.sh and process_results.sh . 


## Reference

If you use or refer to our work, please cite:

@article{Alhama2019cogsci,

  author  = {Raquel G. Alhama and Noam Siegelman and Ram Frost and Blair C. Armstrong},

  title   = {The Role of Information in Visual Word Recognition: A Perceptually-Constrained Connectionist Account},

  journal = {Proceedings of the 41st Annual Conference of the Cognitive Science Society},

  year    = {2019}
}
