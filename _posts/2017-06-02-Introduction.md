---
title:  “Introduction”
date:   2017-06-01 22:37:00
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

In this project, we discuss methods which can be used to detect duplicate questions using Quora dataset. Of course, these methods can be used for other similar datasets.

Methods discussed in this article range from simple TF-IDF, Singular Value Decomposition, Fuzzy Features, Word2Vec features, GloVe features, LSTMs and 1D CNN. We provide a comparison of performance of these algorithms on the Quora dataset.
