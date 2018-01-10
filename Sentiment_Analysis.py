from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd


def sentiment(blob):
    blob = TextBlob("force fuck")

    words = blob.words
    words.lower()

    sentences = blob.sentences

    # sentiment counters
    totalSen = 0
    toxic = 0
    severe_toxic = 0
    obscene = 0
    threat = 0
    insult = 0
    identity_hate = 0

    with open("/Users/massimodaul/PycharmProjects/New/titanic/toxic") as f:
        Toxicwords = f.read().splitlines()

    with open("/Users/massimodaul/PycharmProjects/New/titanic/severe_toxic") as f:
        SevereToxicWords = f.read().splitlines()

    with open("/Users/massimodaul/PycharmProjects/New/titanic/obscene") as f:
        ObsceneWords = f.read().splitlines()

    with open("/Users/massimodaul/PycharmProjects/New/titanic/threat") as f:
        ThreatWords = f.read().splitlines()

    with open("/Users/massimodaul/PycharmProjects/New/titanic/insult") as f:
        InsultWords = f.read().splitlines()

    with open("/Users/massimodaul/PycharmProjects/New/titanic/identity_hate") as f:
        IdentityHateWords = f.read().splitlines()

    for i in range(0, len(words)):
        if words[i] in Toxicwords:
            toxic += 1

        if words[i] in SevereToxicWords:
            severe_toxic += 1

        if words[i] in ObsceneWords:
            obscene += 1

        if words[i] in ThreatWords:
            threat += 1

        if words[i] in InsultWords:
            insult += 1

        if words[i] in IdentityHateWords:
            identity_hate += 1

    for i in range(0, len(sentences)):
        if sentences[i] in Toxicwords:
            toxic += 1

        if sentences[i] in SevereToxicWords:
            severe_toxic += 1

        if sentences[i] in ObsceneWords:
            obscene += 1

        if sentences[i] in ThreatWords:
            threat += 1

        if sentences[i] in InsultWords:
            insult += 1

        if sentences[i] in IdentityHateWords:
            identity_hate += 1

        return toxic, severe_toxic, obscene, threat, insult, identity_hate















