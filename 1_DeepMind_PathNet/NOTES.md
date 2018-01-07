---
_Title_: PathNet: Evolution Channels Gradient Descent in Super Neural Networks

_Author(s)_: Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha†, Andrei A. Rusu, Alexander Pritzel, Daan Wierstra; Google DeepMind

_Keywords_: giant networks, path evolution algorithm, evolution and learning, continual learning, transfer learning, multitask learning, basal ganglia

_Start_: 6/28/17

_End_:
---

1: DeepMind PathNet
==============================

## 📃 Summary
PathNet is a new deep learning architecture that combines modular deep learning, meta-learning, and reinforcement learning. From the paper:
> For artificial general intelligence (AGI) it would be efficient if multiple users trained the same giant neural network, permitting parameter reuse, without catastrophic forgetting. PathNet is a first step in this direction. It is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks.  

## 📝 Notes
- Reuses a network consisting of many networks on multiple tasks
	- Layers of neural networks interconnected through different search methods
- Includes aspects of transfer learning, continual learning, and multitask learning. This allows the network to (in theory) continuously adapt.
- Conditional Logic
	- Programming
		- Computation
		- Conditional logic
		- Iteration/recursion
	- Neuron
		- Computational unit (sum of products)
		- Conditional unit (activation function)
		- Layers added which led to deep learning
	- Add loops into network
		- Led to RNNs
		- RNNs are chaotic so memory (buffer) is needed with led to LSTM  

## 🔎 Research Method
1. Read the introduction
	☑️
2. Identify the big question
	Is it possible to create a neural network algorithm that identifies which parts of the network can be re-used for new tasks and achieve better results than fine-tuning and other hyperparameter optimization?
3. Summarize the background in five sentences or less
	Neural networks, in general, are trained on data for each specific task they are trying to achieve. This is time consuming and not efficient. Transfer learning was developed to bypass this problem but has limited use. PathNet seeks to combine transfer learning, continual learning, and multitask learning to solve the problem of catastrophic forgetting.
4. Identify specific questions
	How can be PathNet be implemented in order to achieve results that are better than fine-tuning and independent learning controls?
5. Identify the approach
	The general approach is as follows:
	> A PathNet is a modular deep neural network having L layers with each layer consisting of M modules. Each mod-ule is itself a neural network, here either convolutional or linear, followed by a transfer function; rectiﬁed linear units are used here. For each layer the outputs of the modules in that layer are summed before being passed into the ac-tive modules of the next layer. A module is active if it is present in the path genotype currently being evaluated (see below). A maximum of N distinct modules per layer are permitted in a pathway (typically N = 3 or 4). The ﬁnal layer is unique and unshared for each task being learned. In the supervised case this is a single linear layer for each task, and in the A3C case each task (e.g. Atari game) has a value function readout and a policy readout.
	The approach for binary MNIST classification, CIFAR and SVHN, Atari games, and Labyrinth games all differed in order to accommodate each task.
6. Read the Methods section and draw a diagram of the experiment
	See figures at the end of the paper.
7. Summarize each result
	- Binary MNIST classification
		\-  
	- CIFAR and SVHN
	- Atari games
	- Labyrinth games
8. Do the results answer the specific questions/goals?
9. Read the conclusion/discussion section
	☑️
10. Read the abstract
	☑️
11. What do others say about this paper?
	Not too much, however, a few articles mentioning it seem to characterize the results in a very positive light.  

## 📗 Resources
- [https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273][1]
- [https://medium.com/intuitionmachine/is-conditional-logic-the-new-deep-learning-hotness-96832774907b][2]
- [https://github.com/jaesik817/pathnet][3]

## 📐 Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-cpd-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizationsbre
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

[1]:	[https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273] "PathNet - A Modular Deep Learning Architecture"
[2]:	https://medium.com/intuitionmachine/is-conditional-logic-the-new-deep-learning-hotness-96832774907b "Is Conditional Logic the New Deep Learning Hotness?"
[3]:	https://github.com/jaesik817/pathnet "PathNet on Github"

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
