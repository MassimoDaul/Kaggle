from Sentiment_Analysis import sentiment, totalsentiment
import NetTest_From_Web
import tensorflow as tf


# update tensorflow to cooroperte with 3.6:
# python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl


comment = "statement to analyze"


# saving sentiment stats as a list.
sentList = (sentiment(comment))

# threshold for comparison

threshold = totalsentiment(comment)


# TODO

"""

Tensorflow tutorial

Either find way to adapt data to work for built neural net or use tensorflow to try to predict if the comment should be 
censored or not


Structure of net:
Neuron1: Toxic counter
Neuron2: Severe_Toxic counter
Neuron3: severe_toxic
Neuron4: obscene
Neuron5: threat
Neuron6: insult
Neuron7: identity_hate

Output:

0 - do not censor

1 - censor

Use data from csv file in repository to get comments.

Manually train, use human judgement to see if comment is appropriate. 
Try to be consistent (Considering other ways of training, that will be more time efficient and correct. 
Hard because if we have something that can do it, we have our job done)

"""



