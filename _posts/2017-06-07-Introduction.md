---
title:  Introduction
date:   2017-06-07 22:37:00
categories: Description
---

**Introduction**


Quora released its first ever dataset publicly on 24th Jan, 2017. This dataset consists of question pairs which are either duplicate or not. Duplicate questions mean the same thing.

For example, the question pairs below are duplicates (from the Quora dataset)

- How does Quora quickly mark questions as needing improvement?
- Why does Quora mark my questions as needing improvement/clarification before I have time to give it details? Literally within seconds


-Why did Trump win the Presidency?
-How did Donald Trump win the 2016 Presidential Election?


-What practical applications might evolve from the discovery of the Higgs Boson?
-What are some practical benefits of discovery of the Higgs Boson?

Some examples of non-duplicate questions are as follows:

-Who should I address my cover letter to if I'm applying for a big company like Mozilla?
-Which car is better from safety view?""swift or grand i10"".My first priority is safety?


-Mr. Robot (TV series): Is Mr. Robot a good representation of real-life hacking and hacking culture? Is the depiction of hacker societies realistic?
-What mistakes are made when depicting hacking in ""Mr. Robot"" compared to real-life cybersecurity breaches or just a regular use of technologies?


-How can I start an online shopping (e-commerce) website?
-Which web technology is best suitable for building a big E-Commerce website?



Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

Here, our goal is to identify which questions asked on Quora, a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.


In this project, we discuss methods which can be used to detect duplicate questions using Quora dataset. Of course, these methods can be used for other similar datasets.

Methods discussed in this article range from simple TF-IDF, Singular Value Decomposition, Fuzzy Features, Word2Vec features, GloVe features, LSTMs and 1D CNN. We provide a comparison of performance of these algorithms on the Quora dataset.


Submission and Evaluation

For each ID in the test set, you must predict the probability that the questions are duplicates (a number between 0 and 1).

Submissions are evaluated on the log loss between the predicted values and the ground truth.
$$l(y,p)= -y \log(p) - (1-y) \log(1-p)$$
The total loss is a finte sum over all examples in the test set: $$\frac{1}{N} \sum_{i \in \texttt{testset}} l(y_i, p_i) $$
