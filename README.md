# Machine Learning Experiments
Compilation of papers, blog posts, repositories, and other experiments from around the Internet.
I put this together mostly to reduce the amount of open tabs I have in Chrome and so that I can
visit whenever I have free time to experiment.

-------------------------------------------------------------------------------

## Machine Learning for Software Engineers
A multi-month study plan for going from a mobile developer to machine learning engineer.

https://github.com/ZuzooVn/machine-learning-for-software-engineers

-------------------------------------------------------------------------------

## Oxford Deep NLP 2017 course
This repository contains the lecture slides and course description for the Deep Natural Language Processing course offered in Hilary Term 2017 at the University of Oxford.

This is an advanced course on natural language processing. Automatically processing natural language inputs and producing language outputs is a key component of Artificial General Intelligence. The ambiguities and noise inherent in human communication render traditional symbolic AI techniques ineffective for representing and analysing language data. Recently statistical techniques based on neural networks have achieved a number of remarkable successes in natural language processing leading to a great deal of commercial and academic interest in the field

This is an applied course focussing on recent advances in analysing and generating speech and text using recurrent neural networks. We introduce the mathematical definitions of the relevant machine learning models and derive their associated optimisation algorithms. The course covers a range of applications of neural networks in NLP including analysing latent dimensions in text, transcribing speech to text, translating between languages, and answering questions. These topics are organised into three high level themes forming a progression from understanding the use of neural networks for sequential language modelling, to understanding their use as conditional language models for transduction tasks, and finally to approaches employing these techniques in combination with other mechanisms for advanced applications. Throughout the course the practical implementation of such models on CPU and GPU hardware is also discussed.

This course is organised by Phil Blunsom and delivered in partnership with the DeepMind Natural Language Research Group.

https://github.com/oxford-cs-deepnlp-2017/lectures

-------------------------------------------------------------------------------

## DeepMind’s PathNet: A Modular Deep Learning Architecture for AGI
PathNet is a new Modular Deep Learning (DL) architecture, brought to you by who else but DeepMind, that highlights the latest trend in DL research to meld Modular Deep Learning, Meta-Learning and Reinforcement Learning into a solution that leads to more capable DL systems. A January 20th, 2017 submitted Arxiv paper “PathNet: Evolution Channels Gradient Descent in Super Neural Networks” (Fernando et. al) has in its abstract the following interesting description of the work:

>For artificial general intelligence (AGI) it would be efficient if multiple users trained the same giant neural network, permitting parameter reuse, without catastrophic forgetting. PathNet is a first step in this direction. It is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks.

https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273

-------------------------------------------------------------------------------

## NSynth: Neural Audio Synthesis
One of the goals of Magenta is to use machine learning to develop new avenues of human expression. And so today we are proud to announce NSynth (Neural Synthesizer), a novel approach to music synthesis designed to aid the creative process.

Unlike a traditional synthesizer which generates audio from hand-designed components like oscillators and wavetables, NSynth uses deep neural networks to generate sounds at the level of individual samples. Learning directly from data, NSynth provides artists with intuitive control over timbre and dynamics and the ability to explore new sounds that would be difficult or impossible to produce with a hand-tuned synthesizer.

The acoustic qualities of the learned instrument depend on both the model used and the available training data, so we are delighted to release improvements to both:

- A dataset of musical notes an order of magnitude larger than other publicly available corpora.
- A novel WaveNet-style autoencoder model that learns codes that meaningfully represent the space of instrument sounds.

https://magenta.tensorflow.org/nsynth

-------------------------------------------------------------------------------

## Market Vectors
In many NLP problems we end up taking a sequence and encoding it into a single fixed size representation, then decoding that representation into another sequence. For example, we might tag entities in the text, translate from English to French or convert audio frequencies to text. There is a torrent of work coming out in these areas and a lot of the results are achieving state of the art performance.

In my mind the biggest difference between the NLP and financial analysis is that language has some guarantee of structure, it’s just that the rules of the structure are vague. Markets, on the other hand, don’t come with a promise of a learnable structure, that such a structure exists is the assumption that this project would prove or disprove (rather it might prove or disprove if I can find that structure).

Assuming the structure is there, the idea of summarizing the current state of the market in the same way we encode the semantics of a paragraph seems plausible to me.

https://github.com/talolard/MarketVectors/blob/master/preparedata.ipynb

-------------------------------------------------------------------------------

## A Visual and Interactive Guide to the Basics of Neural Networks
I’m not a machine learning expert. I’m a software engineer by training and I’ve had little interaction with AI. I had always wanted to delve deeper into machine learning, but never really found my “in”. That’s why when Google open sourced TensorFlow in November 2015, I got super excited and knew it was time to jump in and start the learning journey. Not to sound dramatic, but to me, it actually felt kind of like Prometheus handing down fire to mankind from the Mount Olympus of machine learning. In the back of my head was the idea that the entire field of Big Data and technologies like Hadoop were vastly accelerated when Google researchers released their Map Reduce paper. This time it’s not a paper – it’s the actual software they use internally after years and years of evolution.

So I started learning what I can about the basics of the topic, and saw the need for gentler resources for people with no experience in the field. This is my attempt at that.

https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/

-------------------------------------------------------------------------------

## Deep Reinforcement Learning: Pong from Pixels
This is a long overdue blog post on Reinforcement Learning (RL). RL is hot! You may have noticed that computers can now automatically learn to play ATARI games (from raw game pixels!), they are beating world champions at Go, simulated quadrupeds are learning to run and leap, and robots are learning how to perform complex manipulation tasks that defy explicit programming. It turns out that all of these advances fall under the umbrella of RL research. I also became interested in RL myself over the last ~year: I worked through Richard Sutton’s book, read through David Silver’s course, watched John Schulmann’s lectures, wrote an RL library in Javascript, over the summer interned at DeepMind working in the DeepRL group, and most recently pitched in a little with the design/development of OpenAI Gym, a new RL benchmarking toolkit. So I’ve certainly been on this funwagon for at least a year but until now I haven’t gotten around to writing up a short post on why RL is a big deal, what it’s about, how it all developed and where it might be going.

https://karpathy.github.io/2016/05/31/rl/

-------------------------------------------------------------------------------

## An introduction to Generative Adversarial Networks (with code in TensorFlow)
There has been a large resurgence of interest in generative models recently (see this blog post by OpenAI for example). These are models that can learn to create data that is similar to data that we give them. The intuition behind this is that if we can get a model to write high-quality news articles for example, then it must have also learned a lot about news articles in general. Or in other words, the model should also have a good internal representation of news articles. We can then hopefully use this representation to help us with other related tasks, such as classifying news articles by topic.

Actually training models to create data like this is not easy, but in recent years a number of methods have started to work quite well. One such promising approach is using Generative Adversarial Networks (GANs). The prominent deep learning researcher and director of AI research at Facebook, Yann LeCun, recently cited GANs as being one of the most important new developments in deep learning:

>There are many interesting recent development in deep learning…The most important one, in my opinion, is adversarial training (also called GAN for Generative Adversarial Networks). This, and the variations that are now being proposed is the most interesting idea in the last 10 years in ML, in my opinion.
 
The rest of this post will describe the GAN formulation in a bit more detail, and provide a brief example (with code in TensorFlow) of using a GAN to solve a toy problem.

http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

-------------------------------------------------------------------------------

## A step-by-step guide to building a simple chess AI
Let’s explore some basic concepts that will help us create a simple chess AI:
- move-generation
- board evaluation
- minimax
- and alpha beta pruning.

At each step, we’ll improve our algorithm with one of these time-tested chess-programming techniques.

https://medium.freecodecamp.com/simple-chess-ai-step-by-step-1d55a9266977

-------------------------------------------------------------------------------

## Deep Photo Style Transfer
This paper introduces a deep-learning approach to photographic style transfer that handles a large variety of image content while faithfully transferring the reference style. Our approach builds upon the recent work on painterly transfer that separates style from the content of an image by considering different layers of a neural network. However, as is, this approach is not suitable for photorealistic style transfer. Even when both the input and reference images are photographs, the output still exhibits distortions reminiscent of a painting. Our contribution is to constrain the transformation from the input to the output to be locally affine in colorspace, and to express this constraint as a custom fully differentiable energy term. We show that this approach successfully suppresses distortion and yields satisfying photorealistic style transfers in a broad variety of scenarios, including transfer of the time of day, weather, season, and artistic edits.

https://github.com/luanfujun/deep-photo-styletransfer?utm_content=buffer39dd6&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

-------------------------------------------------------------------------------

## The Unreasonable Effectiveness of Recurrent Neural Networks
There’s something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I’ve in fact reached the opposite conclusion). Fast forward about a year: I’m training RNNs all the time and I’ve witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you.

https://karpathy.github.io/2015/05/21/rnn-effectiveness/

-------------------------------------------------------------------------------

## Machine Learning is Fun! An Introduction to Machine Learning
This guide is for anyone who is curious about machine learning but has no idea where to start. I imagine there are a lot of people who tried reading the wikipedia article, got frustrated and gave up wishing someone would just give them a high-level explanation. That’s what this is.

The goal is be accessible to anyone — which means that there’s a lot of generalizations. But who cares? If this gets anyone more interested in ML, then mission accomplished.

https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471

-------------------------------------------------------------------------------

## Learning AI if You Suck at Math — P5 — Deep Learning and Convolutional Neural Nets in Plain English!
Today, we’re going to write our own Python image recognition program.

To do that, we’ll explore a powerful deep learning architecture called a deep convolutional neural network (DCNN).

https://hackernoon.com/learning-ai-if-you-suck-at-math-p5-deep-learning-and-convolutional-neural-nets-in-plain-english-cda79679bbe3

-------------------------------------------------------------------------------

## Caption this, with Tensorflow
In this article, we will walk through an intermediate-level tutorial on how to train an image caption generator on the Flickr30k data set using an adaptation of Google’s Show and Tell model. We use the TensorFlow framework to construct, train, and test our model because it’s relatively easy to use and has a growing online community.

https://www.oreilly.com/learning/caption-this-with-tensorflow

-------------------------------------------------------------------------------

## Big Picture Machine Learning: Classifying Text with Neural Networks and TensorFlow
In this article, we’ll create a machine learning model to classify texts into categories. We’ll cover the following topics:
1. How TensorFlow works
2. What is a machine learning model
3. What is a Neural Network
4. How the Neural Network learns
5. How to manipulate data and pass it to the Neural Network inputs
6. How to run the model and get the prediction results

https://medium.freecodecamp.com/big-picture-machine-learning-classifying-text-with-neural-networks-and-tensorflow-d94036ac2274

-------------------------------------------------------------------------------

## Recursive Neural Networks with PyTorch
This post walks through the PyTorch implementation of a recursive neural network with a recurrent tracker and TreeLSTM nodes, also known as SPINN—an example of a deep learning model from natural language processing that is difficult to build in many popular frameworks. The implementation I describe is also partially batched, so it’s able to take advantage of GPU acceleration to run significantly faster than versions that don’t use batching.

https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/

-------------------------------------------------------------------------------

## Best Practices for Applying Deep Learning to Novel Applications
This report is targeted to groups who are subject matter experts in their application but deep learning novices. It contains practical advice for those interested in testing the use of deep neural networks on applications that are novel for deep learning. We suggest making your project more manageable by dividing it into phases. For each phase this report contains numerous recommendations and insights to assist novice practitioners.

https://arxiv.org/ftp/arxiv/papers/1704/1704.01568.pdf

-------------------------------------------------------------------------------

## Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning
In this post we will take you behind the scenes on how we built a state-of-the-art Optical Character Recognition (OCR) pipeline for our mobile document scanner. We used computer vision and deep learning advances such as bi-directional Long Short Term Memory (LSTMs), Connectionist Temporal Classification (CTC), convolutional neural nets (CNNs), and more. In addition, we will also dive deep into what it took to actually make our OCR pipeline production-ready at Dropbox scale.

https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/

-------------------------------------------------------------------------------

## EmojiIntelligence
Do you want to teach your machine emojis? 😏 

I created a neural network entirely in Swift. This is a demo to demonstrate what is possible to solve. 

https://github.com/Luubra/EmojiIntelligence?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue

-------------------------------------------------------------------------------

## BEGAN: Boundary Equilibrium Generative Adversarial Networks
Implementation of Google Brain's BEGAN: Boundary Equilibrium Generative Adversarial Networks in Tensorflow. 

BEGAN is the state of the art when it comes to generate realistic faces.

https://github.com/Heumi/BEGAN-tensorflow?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue

-------------------------------------------------------------------------------

## Medical Image Analysis with Deep Learning — I
Analyzing images and videos, and using them in various applications such as self driven cars, drones etc. with underlying deep learning techniques has been the new research frontier. The recent research papers such as “A Neural Algorithm of Artistic Style”, show how a styles can be transferred from an artist and applied to an image, to create a new image. Other papers such as “Generative Adversarial Networks” (GAN) and “Wasserstein GAN” have paved the path to develop models that can learn to create data that is similar to data that we give them. Thus opening up the world to semi-supervised learning and paving the path to a future of unsupervised learning.
While these research areas are still on the generic images, our goal is to use these research into medical images to help healthcare. We need to start with some basics. In this article, I start with basics of image processing, basics of medical image format data and visualize some medical data. In the next article I will deep dive into some convolutional neural nets and use them with Keras for predicting lung cancer.

https://medium.com/@taposhdr/medical-image-analysis-with-deep-learning-i-23d518abf531

-------------------------------------------------------------------------------

## Kalman and Bayesian Filters in Python
Kalman Filter book using Jupyter Notebook. Focuses on building intuition and experience, not formal proofs. Includes Kalman filters,extended Kalman filters, unscented Kalman filters, particle filters, and more. All exercises include solutions.

https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

-------------------------------------------------------------------------------

## PyTorch Playground
Base pretrained models and datasets in pytorch (MNIST, SVHN, CIFAR10, CIFAR100, STL10, AlexNet, VGG16, VGG19, ResNet, Inception, SqueezeNet).

https://github.com/aaron-xichen/pytorch-playground

-------------------------------------------------------------------------------

## Building AnswerBot with Keras and TensorFlow
With the recent advances into neural networks capabilities to process text and audio data we are very close creating a natural human assistant. TensorFlow from Google is one of the most popular neural network library, and using Keras you can simplify TensorFlow usage. TensorFlow brings amazing capabilities into natural language processing (NLP) and using deep learning, we are expecting bots to become even more smarter, closer to human experience. In this technical discussion, we will explore NLP methods in TensorFlow with Keras to create answer bot, ready to answers specific technical questions. You will learn how to use TensorFlow to train an answer bot, with specific technical questions and use various AWS services to deploy answer bot in cloud.

https://github.com/Avkash/mldl/tree/master/tensorbeat-answerbot

-------------------------------------------------------------------------------

## 6.S094: Deep Learning for Self-Driving Cars
This class is an introduction to the practice of deep learning through the applied theme of building a self-driving car. It is open to beginners and is designed for those who are new to machine learning, but it can also benefit advanced researchers in the field looking for a practical overview of deep learning methods and their application.

http://selfdrivingcars.mit.edu

-------------------------------------------------------------------------------

## Dask-SearchCV: Distributed hyperparameter optimization with Scikit-Learn
Last summer I spent some time experimenting with combining dask and scikit-learn (chronicled in this series of blog posts). The library that work produced was extremely alpha, and nothing really came out of it. Recently I picked this work up again, and am happy to say that we now have something I can be happy with. This involved a few major changes:

- A sharp reduction in scope. The previous rendition tried to implement both model and data parallelism. Not being a machine-learning expert, the data parallelism was implemented in a less-than-rigorous manner. The scope is now pared back to just implementing hyperparameter searches (model parallelism), which is something we can do well.
- Optimized graph building. Turns out when people are given the option to run grid search across a cluster, they immediately want to scale up the grid size. At the cost of more complicated code, we can handle extremely large grids (e.g. 500,000 candidates now takes seconds for the graph to build, as opposed to minutes before). It should be noted that for grids this size, an active search may perform significantly better. Relevant issue: # 29.
- Increased compatibility with Scikit-Learn. Now with only a few exceptions, the implementations of GridSearchCV and RandomizedSearchCV should be drop-ins for their scikit-learn counterparts.

http://www.kdnuggets.com/2017/05/dask-searchcv-distributed-hyperparameter-optimization-scikit-learn.html?utm_content=buffer489b5&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

-------------------------------------------------------------------------------

## Kernelized Sorting
Object matching is a fundamental operation in data analysis. It typically requires the definition of a similarity measure between the classes of objects to be matched. Instead, we develop an approach which is able to perform matching by requiring a similarity measure only within each of the classes. This is achieved by maximizing the dependency between matched pairs of observations by means of the Hilbert Schmidt Independence Criterion. This problem can be cast as one of maximizing a quadratic assignment problem with special structure and we present a simple algorithm for finding a locally optimal solution.

http://users.sussex.ac.uk/%7Enq28/kernelized_sorting.html

-------------------------------------------------------------------------------

## Quick Draw! The Data
Over 15 million players have contributed millions of drawings playing Quick, Draw! These doodles are a unique data set that can help developers train new neural networks, help researchers see patterns in how people around the world draw, and help artists create things we haven’t begun to think of. That’s why we’re open-sourcing them, for anyone to play with.

https://quickdraw.withgoogle.com/data

-------------------------------------------------------------------------------

## ML Algorithms
Minimal and clean examples of machine learning algorithms.

https://github.com/rushter/MLAlgorithms

-------------------------------------------------------------------------------

## Face Classification and Detection
Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV.

https://github.com/oarriaga/face_classification

-------------------------------------------------------------------------------

## Convolutional Methods for Text
- RNNS work great for text but convolutions can do it faster
- Any part of a sentence can influence the semantics of a word. For that reason we want our network to see the entire input at once
- Getting that big a receptive can make gradients vanish and our networks fail
- We can solve the vanishing gradient problem with DenseNets or Dilated Convolutions
- Sometimes we need to generate text. We can use “deconvolutions” to generate arbitrarily long outputs.

https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f

-------------------------------------------------------------------------------

## Getting started with ARKit on iOS 11
Apple’s new ARKit APIs make it possible to build compelling augmented reality apps that can run on iOS devices with A9 or A10 processors.

The potential to build the next Pokemon Go has many folk without much iOS/Unity development experience looking to get started — and while Apple’s new capabilities are truly impressive and the WWDC session on the APIs is very informative, the written documentation for ARKit is very sparse.

To help others get to experimenting, this guide walks through installing the iOS 11 beta on your phone or tablet, building a basic ARKit demo, and setting up Unity to build more advanced apps and games.

https://github.com/kylebrussell/ARoniOS/wiki/Getting-started-with-ARKit-on-iOS-11

-------------------------------------------------------------------------------

## How I Built a Reverse Image Search with Machine Learning and TensorFlow
I wanted to write up an end-to-end description of what it’s like to build a machine learning app, and more specifically, how to make your own reverse image search. For this demo, the work is ⅓ data munging/setup, ⅓ model development and ⅓ app development.

At a high-level, I use TensorFlow to create an autoencoder, train it on a bunch of images, use the trained model to find related images, and display them with a Flask app.

In this first post, I’m going to go over my environment and project setup and do a little bit of scaffolding. Ready? Let’s get started.

https://www.codementor.io/jimmfleming/how-i-built-a-reverse-image-search-with-machine-learning-and-tensorflow-part-1-8dje8gjm9

-------------------------------------------------------------------------------

## iOS 11: Machine Learning for everyone

Machine learning in iOS 11 and macOS 10.13

http://machinethink.net/blog/ios-11-machine-learning-for-everyone/

-------------------------------------------------------------------------------

## Supercharge your Computer Vision models with the TensorFlow Object Detection API


At Google, we develop flexible state-of-the-art machine learning (ML) systems for computer vision that not only can be used to improve our products and services, but also spur progress in the research community. Creating accurate ML models capable of localizing and identifying multiple objects in a single image remains a core challenge in the field, and we invest a significant amount of time training and experimenting with these systems.

At Google, we develop flexible state-of-the-art machine learning (ML) systems for computer vision that not only can be used to improve our products and services, but also spur progress in the research community. Creating accurate ML models capable of localizing and identifying multiple objects in a single image remains a core challenge in the field, and we invest a significant amount of time training and experimenting with these systems.

https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html

-------------------------------------------------------------------------------

## Data Scientist Resume Projects

Data scientists are one of the most hirable specialists today, but it’s not so easy to enter this profession without a “Projects” field in your resume. You need experience to get the job, and you need the job to get the experience. Seems like a vicious circle, right?

Statsbot’s data scientist Denis Semenenko wrote this article to help everyone with making the first simple, but yet illustrative data science projects which can take less than a week of work time.

The great advantage of these projects is that each of them is a full-stack data science problem.

This means that you need to formulate the problem, design the solution, find the data, master the technology, build a machine learning model, evaluate the quality, and maybe wrap it into a simple UI. This is a more diverse approach than, for example, Kaggle competition or Coursera lessons (but they are quite good too!).

Keep reading if you want to improve your CV by using a data science project, find ideas for a university project, or just practice in a particular domain of machine learning.

https://blog.statsbot.co/data-scientist-resume-projects-806a74388ae6

-------------------------------------------------------------------------------

## Database of Structural Propensities of Proteins
dspp-keras is a Keras integration for Database of Structural Propensities of Proteins, which provides amino acid sequences of 7200+ unrelated proteins with their propensities to form secondary structures or stay disordered.

https://github.com/PeptoneInc/dspp-keras

-------------------------------------------------------------------------------

## Artificial Intelligence Complete Lectures
Prof. Patrick Henry Winston introduces students to the basic knowledge representation, problem solving, and learning methods of artificial intelligence. Upon completion of this course, students should be able to develop intelligent systems by assembling solutions to concrete computational problems; understand the role of knowledge representation, problem solving, and learning in intelligent-system engineering; and appreciate the role of problem solving, vision, and language in understanding human intelligence from a computational perspective.

http://artificialbrain.xyz/artificial-intelligence-complete-lectures-01-23/
## Keras Visualization
keras-vis is a high-level toolkit for visualizing and debugging your trained keras neural net models. Currently supported visualizations include:

- Activation maximization
- Saliency maps
- Class activation maps
- All visualizations by default support N-dimensional image inputs. i.e., it generalizes to N-dim image inputs to your model.

The toolkit generalizes all of the above as energy minimization problems with a clean, easy to use, and extendable interface. Compatible with both theano and tensorflow backends with 'channels_first', 'channels_last' data format.

https://github.com/raghakot/keras-vis

-------------------------------------------------------------------------------

## An Overview of Multi-task Learning in Deep Neural Networks
In Machine Learning (ML), we typically care about optimizing for a particular metric, whether this is a score on a certain benchmark or a business KPI. In order to do this, we generally train a single model or an ensemble of models to perform our desired task. We then fine-tune and tweak these models until their performance no longer increases. While we can generally achieve acceptable performance this way, by being laser-focused on our single task, we ignore information that might help us do even better on the metric we care about. Specifically, this information comes from the training signals of related tasks. By sharing representations between related tasks, we can enable our model to generalize better on our original task. This approach is called Multi-Task Learning (MTL) and will be the topic of this blog post.

http://sebastianruder.com/multi-task/

-------------------------------------------------------------------------------

## Style Transfer
Implementation of original style transfer paper (https://arxiv.org/abs/1508.06576).

https://github.com/slavivanov/Style-Tranfer

-------------------------------------------------------------------------------

## Not Hot Dog Classifier
Do you watch HBO's silicon valley? Because I do and I was inspired by Mr. Jian-Yang to make my own not hotdog classifier

"What would you say if I told you there is a app on the market that tell you if you have a hotdog or not a hotdog. It is very good and I do not want to work on it any more. You can hire someone else." - Jian-Yang , 2017

https://github.com/kmather73/NotHotdog-Classifier

-------------------------------------------------------------------------------

## Using convolutional neural nets to detect facial keypoints tutorial
This is a hands-on tutorial on deep learning. Step by step, we'll go about building a solution for the Facial Keypoint Detection Kaggle challenge. The tutorial introduces Lasagne, a new library for building neural networks with Python and Theano. We'll use Lasagne to implement a couple of network architectures, talk about data augmentation, dropout, the importance of momentum, and pre-training. Some of these methods will help us improve our results quite a bit.

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

-------------------------------------------------------------------------------

## Deep Learning Scaling is Predictable, Empirically

Deep learning (DL) creates impactful advances following a virtuous recipe: model architecture search, creating large training data sets, and scaling computation. It is widely believed that growing training sets and models should improve accuracy and result in better products. As DL application domains grow, we would like a deeper understanding of the relationships between training set size, computational scale, and model accuracy improvements to advance the state-of-the-art. 

https://arxiv.org/abs/1712.00409

-------------------------------------------------------------------------------

## Welcoming the Era of Deep Neuroevolution

In the field of deep learning, deep neural networks (DNNs) with many layers and millions of connections are now trained routinely through stochastic gradient descent (SGD). Many assume that the ability of SGD to efficiently compute gradients is essential to this capability. However, we are releasing a suite of five papers that support the emerging realization that neuroevolution, where neural networks are optimized through evolutionary algorithms, is also an effective method to train deep neural networks for reinforcement learning (RL) problems. Uber has a multitude of areas where machine learning can improve its operations, and developing a broad range of powerful learning approaches that includes neuroevolution will help us achieve our mission of developing safer and more reliable transportation solutions.

https://eng.uber.com/deep-neuroevolution/

-------------------------------------------------------------------------------

## wav2letter

wav2letter is a simple and efficient end-to-end Automatic Speech Recognition (ASR) system from Facebook AI Research. The original authors of this implementation are Ronan Collobert, Christian Puhrsch, Gabriel Synnaeve, Neil Zeghidour, and Vitaliy Liptchinsky.

https://github.com/facebookresearch/wav2letter

-------------------------------------------------------------------------------

## How To Create Data Products That Are Magical Using Sequence-to-Sequence Models

A tutorial on how to summarize text and generate features from Github Issues using deep learning with Keras and TensorFlow.

https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8

-------------------------------------------------------------------------------

## Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial

In this tutorial, you will learn how to train and test an end-to-end deep learning model for autonomous driving using data collected from the AirSim simulation environment. You will train a model to learn how to steer a car through a portion of the Mountain/Landscape map in AirSim using a single front facing webcam for visual input. Such a task is usually considered the "hello world" of autonomous driving, but after finishing this tutorial you will have enough background to start exploring new ideas on your own. Through the length of this tutorial, you will also learn some practical aspects and nuances of working with end-to-end deep learning methods.

https://github.com/Microsoft/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning

-------------------------------------------------------------------------------

## Cryptocurrency Data Analysis Part I: Obtaining and Playing with Data of Digital Assets

The word “cryptocurrency” has taken the financial world by storm, and yet there is a lack of formal and open research being conducted on the data of the digital assets. Personally, being a cryptocurrency investor and a data scientist, I am fascinated by studying this nascent asset class under the microscope of data analysis and machine learning tools in order to guide my investment decisions.
These series of tutorials will hopefully bridge the gap between data scientists and realm of cryptocurrency; vice versa, non technical crypto traders will be able to use this as an opportunity to acquire some directly applicable coding skills. Our data will come from Poloniex.

https://medium.com/@eliquinox/cryptocurrency-data-analysis-part-i-obtaining-and-playing-with-data-of-digital-assets-2a963a72703b

-------------------------------------------------------------------------------

## Build your own self driving (toy) car

We’ll take Deep Neural Network described in my Behavior Cloning project from Udacity Self Driving Car nano degree course and run it on a remote controlled (RC) race car using Robotic Operating System (ROS) as a middle-ware.

https://towardsdatascience.com/build-your-own-self-driving-toy-car-ad00a6804b53

-------------------------------------------------------------------------------

## Over 150 of the Best Machine Learning, NLP, and Python Tutorials I’ve Found

To help others that are going through a similar discovery process, I’ve put together a list of the best tutorial content that I’ve found so far. It’s by no means an exhaustive list of every ML-related tutorial on the web — that would be overwhelming and duplicative. Plus, there is a bunch of mediocre content out there. My goal was to link to the best tutorials I found on the important subtopics within machine learning and NLP.

https://unsupervisedmethods.com/over-150-of-the-best-machine-learning-nlp-and-python-tutorials-ive-found-ffce2939bd78

-------------------------------------------------------------------------------

## Tensorlayer

TensorLayer is a deep learning and reinforcement learning library based on TensorFlow. It provides rich data processing, model training and serving modules to help both researchers and engineers build practical machine learning workflows.

https://github.com/tensorlayer/tensorlayer

-------------------------------------------------------------------------------

## europilot

Europilot is an open source project that leverages the popular Euro Truck Simulator(ETS2) to develop self-driving algorithms.

https://github.com/marsauto/europilot

-------------------------------------------------------------------------------

## Understanding and Implementing CycleGAN in TensorFlow

The paper we are going to implement is titled "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". The title is quite a mouthful and it helps to look at each phrase individually before trying to understand the model all at once.

https://hardikbansal.github.io/CycleGANBlog/

-------------------------------------------------------------------------------

## Real-time object detection with YOLO

In this blog post I’ll describe what it took to get the “tiny” version of YOLOv2 running on iOS using Metal Performance Shaders.

http://machinethink.net/blog/object-detection-with-yolo/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue

-------------------------------------------------------------------------------

## Exploring LSTMs

The first time I learned about LSTMs, my eyes glazed over.

Not in a good, jelly donut kind of way.

It turns out LSTMs are a fairly simple extension to neural networks, and they're behind a lot of the amazing achievements deep learning has made in the past few years. So I'll try to present them as intuitively as possible – in such a way that you could have discovered them yourself.

http://blog.echen.me/2017/05/30/exploring-lstms/

-------------------------------------------------------------------------------

## Deep Learning Is Not Good Enough, We Need Bayesian Deep Learning for Safe AI

Understanding what a model does not know is a critical part of many machine learning systems. Unfortunately, today’s deep learning algorithms are usually unable to understand their uncertainty. 

https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/

-------------------------------------------------------------------------------

## A 2017 Guide to Semantic Segmentation with Deep Learning

In this post, I review the literature on semantic segmentation. Most research on semantic segmentation use natural/real world image datasets. Although the results are not directly applicable to medical images, I review these papers because research on the natural images is much more mature than that of medical images.

http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Interpreting neurons in an LSTM network

A few months ago, we showed how effectively an LSTM network can perform text transliteration.

For humans, transliteration is a relatively easy and interpretable task, so it’s a good task for interpreting what the network is doing, and whether it is similar to how humans approach the same task.

In this post we’ll try to understand: What do individual neurons of the network actually learn? How are they used to make decisions?

http://yerevann.github.io/2017/06/27/interpreting-neurons-in-an-LSTM-network/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Neural Machine Translation (seq2seq) Tutorial

Sequence-to-sequence (seq2seq) models (Sutskever et al., 2014, Cho et al., 2014) have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization. This tutorial gives readers a full understanding of seq2seq models and shows how to build a competitive seq2seq model from scratch. We focus on the task of Neural Machine Translation (NMT) which was the very first testbed for seq2seq models with wild success. 

https://github.com/tensorflow/nmt

-------------------------------------------------------------------------------

## Learning the Enigma with Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are Turing-complete. In other words, they can approximate any function. As a tip of the hat to Alan Turing, let’s see if we can use them to learn the Nazi Enigma.

https://greydanus.github.io/2017/01/07/enigma-rnn/

-------------------------------------------------------------------------------

## 100 days of algorithms

I set the challenge for myself to implement algorithm by algorithm, day by day, until the number reaches 100.

https://medium.com/100-days-of-algorithms/

https://github.com/coells/100days

-------------------------------------------------------------------------------

## An end to end implementation of a Machine Learning pipeline

As a researcher on Computer Vision, I come across new blogs and tutorials on ML (Machine Learning) every day. However, most of them are just focussing on introducing the syntax and the terminology relavant to the field. For example - a 15 minute tutorial on Tensorflow using MNIST dataset, or a 10 minute intro to Deep Learning in Keras on Imagenet.

While people are able to copy paste and run the code in these tutorials and feel that working in ML is really not that hard, it doesn't help them at all in using ML for their own purposes. For example, they never introduce you to how you can run the same algorithm on your own dataset. Or, how do you get the dataset if you want to solve a problem. Or, which algorithms do you use - Conventional ML, or Deep Learning? How do you evaluate your models performance? How do you write your own model, as opposed to choosing a ready made architecture? All these form fundamental steps in any Machine Learning pipeline, and it is these steps that take most of our time as ML practitioners.

This tutorial breaks down the whole pipeline, and leads the reader through it step by step in an hope to empower you to actually use ML, and not just feel that it was not too hard. Needless to say, this will take much longer than 15-30 minutes. I believe a weekend would be a good enough estimate.

https://spandan-madan.github.io/DeepLearningProject/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Reinforcement learning for complex goals, using TensorFlow

Reinforcement learning (RL) is about training agents to complete tasks. We typically think of this as being able to accomplish some goal. Take, for example, a robot we might want to train to open a door. Reinforcement learning can be used as a framework for teaching the robot to open the door by allowing it to learn from trial and error. But what if we are interested in having our agent solve not just one goal, but a set that might vary over time?

In this article, and the accompanying notebook available on GitHub, I am going to introduce and walk through both the traditional reinforcement learning paradigm in machine learning as well as a new and emerging paradigm for extending reinforcement learning to allow for complex goals that vary over time.

https://www.oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Building a Music Recommender with Deep Learning

Wouldn’t it be cool if you could discover music that was released a few years ago that sounds similar to a new song that you like? Surely Juno are missing out on potential sales by not offering this type of feature on their website.

After being inspired by a blog post I’d read recently from somebody who had classified music genres for songs in their own music library, I decided to see if I could adapt that methodology to build a music recommender.

http://mattmurray.net/building-a-music-recommender-with-deep-learning/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Deep Learning - The Straight Dope

This repo contains an incremental sequence of notebooks designed to teach deep learning, Apache MXNet (incubating), and the gluon interface. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and code together in one place. If we’re successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge there’s no source out there that teaches either (1) the full breadth of concepts in modern deep learning or (2) interleaves an engaging textbook with runnable code. We’ll find out by the end of this venture whether or not that void exists for a good reason.

http://gluon.mxnet.io/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly

-------------------------------------------------------------------------------

## Yes you should understand backprop

When we offered CS231n (Deep Learning class) at Stanford, we intentionally designed the programming assignments to include explicit calculations involved in backpropagation on the lowest level. The students had to implement the forward and the backward pass of each layer in raw numpy. Inevitably, some students complained on the class message boards:

“Why do we have to write the backward pass when frameworks in the real world, such as TensorFlow, compute them for you automatically?”

This is seemingly a perfectly sensible appeal - if you’re never going to write backward passes once the class is over, why practice writing them? Are we just torturing the students for our own amusement? Some easy answers could make arguments along the lines of “it’s worth knowing what’s under the hood as an intellectual curiosity”, or perhaps “you might want to improve on the core algorithm later”, but there is a much stronger and practical argument, which I wanted to devote a whole post to:

> The problem with Backpropagation is that it is a leaky abstraction.

https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

-------------------------------------------------------------------------------

## Using neural nets to recognize handwritten digits

Neural Networks and Deep Learning is a free online book. The book will teach you about:

- Neural networks, a beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data

- Deep learning, a powerful set of techniques for learning in neural networks

Neural networks and deep learning currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing. This book will teach you many of the core concepts behind neural networks and deep learning.

http://neuralnetworksanddeeplearning.com/chap1.html

-------------------------------------------------------------------------------

## Visual Information Theory

I love the feeling of having a new way to think about the world. I especially love when there’s some vague idea that gets formalized into a concrete concept. Information theory is a prime example of this.

Information theory gives us precise language for describing a lot of things. How uncertain am I? How much does knowing the answer to question A tell me about the answer to question B? How similar is one set of beliefs to another? I’ve had informal versions of these ideas since I was a young child, but information theory crystallizes them into precise, powerful ideas. These ideas have an enormous variety of applications, from the compression of data, to quantum physics, to machine learning, and vast fields in between.

Unfortunately, information theory can seem kind of intimidating. I don’t think there’s any reason it should be. In fact, many core ideas can be explained completely visually!

http://colah.github.io/posts/2015-09-Visual-Information/

-------------------------------------------------------------------------------

## Constrained Policy Optimization

Deep reinforcement learning (RL) has enabled some remarkable achievements in hard control problems: with deep RL, agents have learned to play video games directly from pixels, to control robots in simulation and in the real world, to learn object manipulation from demonstrations, and even to beat human grandmasters at Go. Hopefully, we’ll soon be able to take deep RL out of the lab and put it into practical, everyday technologies, like UAV control and household robots. But before we can do that, we have to address the most important concern: safety.

http://bair.berkeley.edu/blog/2017/07/06/cpo/

-------------------------------------------------------------------------------

## An Empirical Study of AI Population Dynamics with Million-agent Reinforcement Learning

In this paper, we conduct an empirical study on discovering the ordered collective dynamics obtained by a population of artificial intelligence (AI) agents. Our intention is to put AI agents into a simulated natural context, and then to understand their induced dynamics at the population level. In particular, we aim to verify if the principles developed in the real world could also be used in understanding an artificially-created intelligent population. To achieve this, we simulate a large-scale predator-prey world, where the laws of the world are designed by only the findings or logical equivalence that have been discovered in nature. We endow the agents with the intelligence based on deep reinforcement learning, and scale the population size up to millions. Our results show that the population dynamics of AI agents, driven only by each agent's individual self interest, reveals an ordered pattern that is similar to the Lotka-Volterra model studied in population biology. We further discover the emergent behaviors of collective adaptations in studying how the agents' grouping behaviors will change with the environmental resources. Both of the two findings could be explained by the self-organization theory in nature.

https://arxiv.org/abs/1709.04511

-------------------------------------------------------------------------------

## Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow

Last month, Uber Engineering introduced Michelangelo, an internal ML-as-a-service platform that democratizes machine learning and makes it easy to build and deploy these systems at scale. In this article, we pull back the curtain on Horovod, an open source component of Michelangelo’s deep learning toolkit which makes it easier to start—and speed up—distributed deep learning projects with TensorFlow.

https://eng.uber.com/horovod/

-------------------------------------------------------------------------------

## Introduction to web scraping with Python

Data is the core of predictive modeling, visualization, and analytics. Unfortunately, the needed data is not always readily available to the user, it is most often unstructured. The biggest source of data is the Internet, and with programming, we can extract and process the data found on the Internet for our use – this is called web scraping. Web scraping allows us to extract data from websites and to do what we please with it. In this post, I will show you how to scrape a website with only a few of lines of code in Python.

https://datawhatnow.com/introduction-web-scraping-python/

-------------------------------------------------------------------------------

## How to Find Wally with a Neural Network

Deep learning provides yet another way to solve the Where’s Wally puzzle problem. But unlike traditional image processing computer vision methods, it works using only a handful of labelled examples that include the location of Wally in an image.

https://towardsdatascience.com/how-to-find-wally-neural-network-eddbb20b0b90

-------------------------------------------------------------------------------

## Using Machine Learning to Predict the Weather: Part 1

This is the first article of a multi-part series on using Python and Machine Learning to build models to predict weather temperatures based off data collected from Weather Underground. The series will be comprised of three different articles describing the major aspects of a Machine Learning project. The topics to be covered are:

1. Data collection and processing (this article)

2. Linear regression models (article 2)

3. Neural network models (article 3)

The data used in this series will be collected from Weather Underground's free tier API web service. I will be using the requests library to interact with the API to pull in weather data since 2015 for the city of Lincoln, Nebraska. Once collected, the data will need to be process and aggregated into a format that is suitable for data analysis, and then cleaned.

http://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/

-------------------------------------------------------------------------------

## Introduction to Gaussian Processes - Part I

Gaussian processes may not be at the center of current machine learning hype but are still used at the forefront of research – they were recently seen automatically tuning the MCTS hyperparameters for AlphaGo Zero for instance. They manage to be very easy to use while providing rich modeling capacity and uncertainty estimates.

However they can be pretty hard to grasp, especially if you’re used to the type of models we see a lot of in deep learning. So hopefully this guide can fix that! It assumes a fairly minimal ML background and I aimed for a more visual & intuitive introduction without totally abandoning the theory. To get the most out of it I recommend downloading the notebook and experimenting with all the code!

http://bridg.land/posts/gaussian-processes-1

-------------------------------------------------------------------------------

## How Docker Can Help You Become A More Effective Data Scientist

For the past 5 years, I have heard lots of buzz about docker containers. It seemed like all my software engineering friends are using them for developing applications. I wanted to figure out how this technology could make me more effective but I found tutorials online either too detailed: elucidating features I would never use as a data scientist, or too shallow: not giving me enough information to help me understand how to be effective with Docker quickly.

https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5

-------------------------------------------------------------------------------

## Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow

In this post we’ll implement a retrieval-based bot. Retrieval-based models have a repository of pre-defined responses they can use, which is unlike generative models that can generate responses they’ve never seen before. A bit more formally, the input to a retrieval-based model is a context c (the conversation up to this point) and a potential response r. The model outputs is a score for the response. To find a good response you would calculate the score for multiple responses and choose the one with the highest score.

http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/

-------------------------------------------------------------------------------

## Colorizing B&W Photos with Neural Networks

Earlier this year, Amir Avni used neural networks to troll the subreddit /r/Colorization - a community where people colorize historical black and white images manually using Photoshop. They were astonished with Amir’s deep learning bot - what could take up to a month of manual labour could now be done in just a few seconds.

I was fascinated by Amir’s neural network, so I reproduced it and documented the process.

https://blog.floydhub.com/colorizing-b&w-photos-with-neural-networks/

-------------------------------------------------------------------------------



-------------------------------------------------------------------------------



-------------------------------------------------------------------------------



-------------------------------------------------------------------------------



-------------------------------------------------------------------------------



-------------------------------------------------------------------------------

