# personalized-hate-speech-recognition


## Abstract
Hate speech, that is, speech that contains elements that humiliate or ridicule others, is unfortunately
becoming more and more common, especially on the Internet. The serious effects it causes to those who are
exposed to it and its growing prevalence have led to a greater need for automated methods to detect it. Due to
the subjective nature of hate speech, personalised methods, which take into account the individual sensitivity
of the user, work exceptionally well in its detection. However, there are currently not many such methods.
This thesis (and repo) proposes a new method for personalised hate speech detection. This new method innovatively
uses the users’ modelled attitudes to different groups of topics, to perform a personalised classification of
potentially aggressive content. The advantage of the proposed method is the easy process of adaptation to the
individual preferences of the users. The method has been tested and compared with other existing methods
for this task on the Wikipedia Talk Labels: Aggression dataset. The results obtained by the new method do
not differ significantly from existing solutions and are definitely better than the generalisation method, which
does not use personalization.

## Method architecture

## Files structure

- src/classification - deep learning classification 

- src/preprocessing:
    - 1_topic_modeling.ipynb -> basic preprocessing + topic modeling + topics analysis
    - 2_clustering.ipynb -> topics clustering + matrix correlation + clustering analysis
    - 3_metrics_and_modeling.ipynb -> user_metrics + logistic regression classification 

