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