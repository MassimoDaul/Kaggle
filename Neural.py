"""

Massimo Daul

Might need bigger computer to run


program to analyze web page comment and determine if it should be censored. Trained net on wikipedia comments from a
kaggle data base.


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

Uses data from csv file in repository to get comments.

"""

from Sentiment_Analysis import sentiment
import NetTest_From_Web
import tensorflow as tf
import pandas as pd
import csv
from numpy import exp, array, random, dot
import numpy
import random
# populating our list of toxic comments using the csv file:

ToxicComments = []
CleanComments = []

with open("/Users/massimodaul/Downloads/train-2.csv", "r") as train_df:
    reader = csv.DictReader(train_df)

    for row in reader:
        if (row['toxic'] == "1" or row['severe_toxic'] == "1" or row['obscene'] == "1" or row['threat'] == "1"
                or row['insult'] == "1" or row['identity_hate'] == "1"):

            ToxicComments.append(row['comment_text'])

with open("/Users/massimodaul/Downloads/train-2.csv", "r") as train_df:
    reader = csv.DictReader(train_df)

    for row in reader:
        if row['toxic'] == "0":
            if row['severe_toxic'] == "0":
                if row['obscene'] == "0":
                    if row["threat"] == "0":
                        if row['insult'] == "0":
                            if row['identity_hate'] == "0":
                                CleanComments.append(row['comment_text'])

# populating sentiment analysis values of the toxic and clean comments for array

ToxicSentiments = []
CleanSentiments = []

for i in range(len(ToxicComments)):
    ToxicSentiments.append(sentiment(ToxicComments[i]))


for i in range(len(CleanComments)):
    CleanSentiments.append(sentiment(CleanComments[i]))


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        numpy.random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * numpy.random.random((7, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(1, number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            for x in range(0, len(self.synaptic_weights)):
                self.synaptic_weights[x] += adjustment[x]

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
        
    # test function checks for [0, 0, 0, 0, 0, 0, 0] => 0.5 error 
    def test(self, TestInput):
        
        out = self.__sigmoid(dot(TestInput, self.synaptic_weights))
        
        if out == 0.5:
            return 0
        
        else:
            return out


# Initialize a single neuron neural network.
neural_network = NeuralNetwork()

print("Random starting synaptic weights: ")
print(neural_network.synaptic_weights)

# very tedious way of preparing input values (NEED TO CHANGE)

# first toxic comment
a = ToxicSentiments[0][0]
b = ToxicSentiments[0][1]
c = ToxicSentiments[0][2]
d = ToxicSentiments[0][3]
e = ToxicSentiments[0][4]
f = ToxicSentiments[0][5]
g = ToxicSentiments[0][6]

# second toxic comment
h = ToxicSentiments[1][0]
i = ToxicSentiments[1][1]
j = ToxicSentiments[1][2]
k = ToxicSentiments[1][3]
l = ToxicSentiments[1][4]
m = ToxicSentiments[1][5]
n = ToxicSentiments[1][6]

# third toxic comment
o = ToxicSentiments[2][0]
p = ToxicSentiments[2][1]
q = ToxicSentiments[2][2]
r = ToxicSentiments[2][3]
s = ToxicSentiments[2][4]
t = ToxicSentiments[2][5]
u = ToxicSentiments[2][6]

# fourth toxic comment
v = ToxicSentiments[3][0]
w = ToxicSentiments[3][1]
x = ToxicSentiments[3][2]
y = ToxicSentiments[3][3]
z = ToxicSentiments[3][4]
a1 = ToxicSentiments[3][5]
b1 = ToxicSentiments[3][6]

# first clean comment
c1 = CleanSentiments[1][0]
d1 = CleanSentiments[1][1]
e1 = CleanSentiments[1][2]
f1 = CleanSentiments[1][3]
g1 = CleanSentiments[1][4]
h1 = CleanSentiments[1][5]
i1 = CleanSentiments[1][6]

# second clean comment
j1 = CleanSentiments[1][0]
k1 = CleanSentiments[1][1]
l1 = CleanSentiments[1][2]
m1 = CleanSentiments[1][3]
n1 = CleanSentiments[1][4]
o1 = CleanSentiments[1][5]
p1 = CleanSentiments[1][6]

# third clean comment
q1 = CleanSentiments[2][0]
r1 = CleanSentiments[2][1]
s1 = CleanSentiments[2][2]
t1 = CleanSentiments[2][3]
u1 = CleanSentiments[2][4]
v1 = CleanSentiments[2][5]
w1 = CleanSentiments[2][6]

# fourth clean comment
x1 = CleanSentiments[3][0]
y1 = CleanSentiments[3][1]
z1 = CleanSentiments[3][2]
a2 = CleanSentiments[3][3]
b2 = CleanSentiments[3][4]
c2 = CleanSentiments[3][5]
d2 = CleanSentiments[3][6]

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value

training_set_inputs = array([[a, b, c, d, e, f, g], [h, i, j, k, l, m, n], 
                             [o, p, q, r, s, t, u], [v, w, x, y, z, a1, b1],
                             [c1, d1, e1, f1, g1, h1, i1], [j1, k1, l1, m1, n1, o1, p1],
                             [q1, r1, s1, t1, u1, v1, w1], [x1, y1, z1, a2, b2, c2, d2]])

training_set_outputs = array([[1, 1, 1, 1, 0, 0, 0, 0]]).T

# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 10000)

print("New synaptic weights after training: ")
print(neural_network.synaptic_weights)

"""

Test the neural network with a new situation. Should have an output of 1. 
Don't need to use variable sentiment values as the array input works fine, the error is with the adjustment in train. 

TODO - Fix adjustment error so that train can except same format as think. 


"""
comment = ToxicSentiments[12]
comment_text = ToxicComments[12]

print("Considering new situation:")
print("Comment:\n\n")
print(comment_text)

NewSituation = neural_network.test(array([numpy.squeeze(comment)]))
print(comment)
print(NewSituation)

if NewSituation > 0.5:
    print("censor")
if NewSituation < 0.5:
    print("do not censor")
