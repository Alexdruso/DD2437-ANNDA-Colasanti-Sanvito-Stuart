Lab 4
=====
	
0. Edit history
---------------

30/09/2019 : README.txt file was updated with instructions to create directories before running the scripts
04/10/2019 : rbm.py and dbn.py was updated. Comments were added with [TODO TASK NUMBER] based on students' comments. Code was not modified.
28/09/2020 : Small bug in util.softmax fixed. Updated code to python3. Empty folders for trained rbm and dbn are created.


1. Files included
-----------------

run.py			    Main file to run all networks.
util.py			    Utility file containing activation functions, sampling methods, load/save files, etc.
rbm.py			    Contains the Restricted Boltzmann Machine class.
dbn.py			    Contains the Deep Belief Network class.

train-images-idx3-ubyte	    MNIST training images
train-labels-idx1-ubyte	    MNIST training labels
t10k-images-idx3-ubyte	    MNIST test images
t10k-labels-idx1-ubyte	    MNIST test labels
trained_rbm	            Directory to store trained RBM model
trained_dbn	            Directory to store trained DBN model


2. Preliminaries
----------------

All the functions and classes needed for the lab are provided as templates, and you will implement them as described in the lab instructions. Before working on the code, it is recommended to execute the run.py and make sure you get the following results.

 > python run.py

 Starting a Restricted Boltzmann Machine..
 learning CD1
 iteration=      0 recon_loss=0.0000
 iteration=   5000 recon_loss=0.0000

 Starting a Deep Belief Net..
 training vis--hid
 training hid--pen
 training pen+lbl--top
 accuracy = 9.87%
 accuracy = 9.80%
 
 training wake-sleep..
 iteration=      0
 accuracy = 9.87%
 accuracy = 9.80%

The first part of run.py creates a RestrictedBoltzmannMachine and runs contrastive divergence learning. The reconstruction loss is printed while learning, and the receptive fields are stored as 'rf.<iteration>.png' in the same directory. Since there is no learning implemented yet, you will see the loss to be zero, and receptive fields as random pixels.

The second part of run.py creates a DeepBeliefNet. Make sure that the RBM from previous part works as expected before starting with DBNs. The learning is by greedy layer-wise stacking of RBMS, and the DBN architecture is by default the model discussed in Hinton, Osindero & Teh (2006) with 3 hidden layers. After training, the network is evaluated and the train/test set accuracy is printed. You will see the accuracy is around 10.0%. The network is then run as generative model, and the results of the network generating each digit (from 0 to 9) are stored as videos 'rbms.generate<digit>.mp4' in the same directory.

The last part of run.py uses the DeepBeliefNet from last part, and fine-tunes the parameters by the wake-sleep algorithm. After training, as in the last part, the train/test set accuracy is printed. The generative model results of each digit (from 0 to 9) are stored as videos 'dbn.generate<digit>.mp4' in the same directory. 

The learning methods load the parameters from file by default. If there are no stored parameters, the learning is implemented. When the greedy learning and wake-sleep learning are successful, all the parameters are stored in the directories 'trained_rbm/' and 'trained_dbn/'. Make sure to clear the directories, if you do not want to reuse the parameters. 

3. Packages for running the code
--------------------------------

Running the code requires python (version 3.8 was test) and the following libraries:

numpy (tested on 1.17.0)
matplotlib.pyplot (tested on matplotlib 3.1.2)
matplotlib.animation (tested on matplotlib 3.1.2 used only for recording videos in the generative mode)
struct (used only for loading mnist IDX files)

numpy and matplotlib.pyplot are essential for running the code. If you do not have matplotlib.animaton and struct, you might have to use other alternatives. struct can be replaced with any other methods to load the IDX formatted binary files. matplotlib.animation can be replaced with other packages to create videos, or you can skip the videos all together and just have a collection of images from the generative model.

4. Implementing the tasks
-------------------------

The tasks in the lab will involve modifying rbm.py and dbn.py. The util.py contains necessary utility functions that are already implemented, and it is recommended that you read through the functions for your tasks. The run.py is a general wrapper which you might also have to edit based on your needs.

TASK 4.1 Here you will create a Restricted Boltzmann Machine, implement the inference rules, and learn the parameters with contrastive divergence. You would have to work on the following functions :

 rbm.cd1()   	   	   contrastive-divergence method for computing the gradients of the parameters
 rbm.update_params()	   updates the parameters from the gradients 
 rbm.get_v_given_h()	   computes the visible probabilities and activations samples given hidden activations
 rbm.get_h_given_v()	   computes the hidden probabilities and activations samples given visible activations
 

TASK 4.2 Here you will create a Deep Belief Net by stacking RBMs, and implement inference rules for the network. Since stacking RBMs on top changes the undirected connections into directed connections, the RBM inference rules no longer hold. You would have to write the new rules, although they are very similar to the old ones. You would have to work on the following functions:

 dbn.train_greedylayerwise()	method for greedy layer-wise stacking of RBMs
 dbn.recognize()		classification method that takes in images and predicts the labels
 dbn.generate()			generation method that takes the labels and generates images
 rbm.get_v_given_h_dir()	rbm method similar to rbm.get_v_given_h but with directed connections
 rbm.get_h_given_v_dir() 	rbm method similar to rbm.get_h_given_v but with directed connections

TASK 4.3 (non-mandatory) Here you will implement the wake-sleep algorithm to fine-tune all the parameters of the DBN. You would also have to implement new RBM methods for learning directed connections. You would have to work on the following functions:

 dbn.train_wakesleep_finetune()	  main method for wake-sleep learning
 rbm.update_generate_params() 	  updates the generative parameters (directed)	
 rbm.update_recognize_params()    updates the recognition parameters (directed)	

More details about each function are provided with the functions templates.

5. Training time
----------------

We give a rough estimate of how much running time can be expected for each section of this lab. Compared to previous labs, this involves learning real world dataset and training large number of parameters, so training time might be considerably longer. The timing below is for training the whole dataset. While developing/debugging code, it might be time-saving to train on smaller training set (~10%) and see favorable results, before training on the whole set.

TASK 4.1 Training time will be in the order 10-20 minutes for the whole training set.
TASK 4.2 Since this involves training three seperate RBMs, it is in the order of three times longer than Task I, 30 to 90 minutes.
TASK 4.3 In the range from 30 to 60 minutes.


