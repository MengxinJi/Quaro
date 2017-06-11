---
title:  Feature Engineering
date:   2017-06-04 22:37:00
categories: Description
---

Number of words/characters in question1

Number of words/characters in question2

Number of common words in question1 and question2/Overlapping score

word2vec based features, where each sentence is converted into a vector so that the different distance measures between the two vectors can be evaluated:

Cosine distance between vectors of question1 and question2

Manhattan distance between vectors of question1 and question2

Euclidean distance between vectors of question1 and question2

Minkowski distance between vectors of question1 and question2

TF-IDF (term-frequency-inverse-document-frequency) based features

Leaky features.

### Feature: Overlapping Score

The following word_match_share function could give us the overlapping score of a pair of questions.

Assume the question pair $q_1$ and $q_2$ are the preprocessed strings after Problem 1, and $q_1 = [w_{11}, w_{12} \cdots, w_{1n}], q2 = [w_{21}, w_{22} \cdots, w_{2m}]$.
(the strings are split into words using white space), then we define the overlapping scores as the number of
overlapping words divided by m + n (sum of number of words). More specifically,
overlapping score is 
$$overlapping score(q_1, q_2) = ( \sum_{1}^n 1( w_{1i} \in q_2) + \sum_{1}^m 1( w_{2i} \in q_1) ) /(m + n) $$

One problem of this feature is that the overlapping score might be dominated by the “stop words”, such as “is”, “a”, “the”, “what”. If all the words in both questions are stop words, we will just report overlapping score to be 0. We use the stop words from the "nltk.corpus".
Now we try to improve the predictive power of our overlapping score by removing the stop words in each question.

From lable distribution of the overlapping score plot, we can see that those duplicate question pairs have a relative higher overlapping score than those non-duplicate question pairs. However, the two distributions still share a lot in common. But anyway, we think this feature might be a useful feature in predicting which question pairs are duplicate. At least, it will be very good at predicting questions which are very different (has a very low overlapping score). And it may not be good at finding questions which are definitely duplicate.


### Feature: TF-IDF

One way to improve this overlapping score feature is to use TF-IDF (term-frequency-inverse-document-frequency). The idea is the same with overlapping score feature, but they give different weights to different words according to how uncommon the words are. Those words that appear frequently should be having a lower weight than those words that appear rarely. This is reasonable because words such as "the" or "of", which are stopping words, appear very frequently without carrying any information. However, words such as "chess" or "AlphaGo" appear less frequently but carry more information. 

