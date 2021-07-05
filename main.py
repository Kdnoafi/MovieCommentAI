import os
import random

positive_corpus = []
negative_corpus = []

dir = 'imdb/pos';
for file in os.listdir(dir):
  path = os.path.join(dir, file)
  f = open(path, 'r')
  positive_corpus.append(f.read())
  f.close()

dir = 'imdb/neg';
for file in os.listdir(dir):
  path = os.path.join(dir, file)
  f = open(path, 'r')
  negative_corpus.append(f.read())
  f.close()

import nltk
nltk.download()

from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import random
import math
import os
import collections

# a)

clean_corpus_pos = []
clean_corpus_neg = []
porter = PorterStemmer()
special = {"<", "/>", "br,", ".<", "/><", "br", "!\"", "!!", "!!!", "!!!!", "!!!!!", "!!!!!!"}

stop_punc = set(stopwords.words('english')).union(set(punctuation).union(special))
for doc in positive_corpus:
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [w for w in words_lower if w not in stop_punc]
    words_stemmed = [porter.stem(w) for w in words_filtered]
    clean_corpus_pos.append(words_stemmed)

for doc in negative_corpus:
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [w for w in words_lower if w not in stop_punc]
    words_stemmed = [porter.stem(w) for w in words_filtered]
    clean_corpus_neg.append(words_stemmed)

clean_corpus = clean_corpus_neg + clean_corpus_pos
random.shuffle(clean_corpus)

brP = 0
brN = 0

nb_train = 2000  # 80%
nb_test = 500  # 20%

vocab_set = set()
for doc in clean_corpus[:nb_train + nb_test]:
    for word in doc:
        vocab_set.add(word)
vocab = list(vocab_set)

print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))
print()

np.set_printoptions(precision=2, linewidth=200)


def occ_score(word, doc):
    return 1 if word in doc else 0


def numocc_score(word, doc):
    return doc.count(word)


def freq_score(word, doc):
    return doc.count(word) / len(doc)


pos_neg_values = [0] * nb_train
test_values = [0] * nb_test


def fill_numocc(X, corpus, vocab, filled):
    newIndex = 0
    if filled == True:
        newIndex = len(clean_corpus)

    for score_fn in [numocc_score]:
        for doc_idx in range(nb_train):
            doc = clean_corpus[doc_idx]

            if doc in clean_corpus_pos:
                pos_neg_values[doc_idx] = 1

            for word_idx in range(len(vocab)):
                word = vocab[word_idx]
                cnt = score_fn(word, doc)
                X[doc_idx + newIndex][word_idx] = cnt


def fill_numocc_no_values(X, corpus, vocab, filled):
    newIndex = 0
    if filled == True:
        newIndex = len(clean_corpus)

    for score_fn in [numocc_score]:
        for doc_idx in range(nb_test):
            doc = clean_corpus[doc_idx + nb_train]

            if doc in clean_corpus_pos:
                test_values[doc_idx] = 1

            for word_idx in range(len(vocab)):
                word = vocab[word_idx]
                cnt = score_fn(word, doc)
                X[doc_idx + newIndex][word_idx] = cnt


def fill_freq(corpus, vocab, filled):
    for score_fn in [freq_score]:
        freq = np.zeros((len(corpus), len(vocab)), dtype=np.float32)
        for doc_idx in range(len(corpus)):
            doc = corpus[doc_idx]
            for word_idx in range(len(vocab)):
                word = vocab[word_idx]
                cnt = score_fn(word, doc)
                freq[doc_idx][word_idx] = cnt
        print('freq:')
        print(freq)
        print()


train_bow = np.zeros((nb_train, len(vocab)), dtype=np.float32)
fill_numocc(train_bow, clean_corpus[:nb_train], vocab, False)
print('Train bow: ')
print(train_bow)
print()

test_bow = np.zeros((nb_test, len(vocab)), dtype=np.float32)
fill_numocc_no_values(test_bow, clean_corpus[nb_train:nb_train + nb_test], vocab, False)
print('Test bow: ')
print(test_bow)
print()


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, alpha):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.alpha = alpha

    def fit(self, X, Y):
        nb_examples = X.shape[0]

        self.priors = np.bincount(Y) / nb_examples
        print('Priors: ')
        print(self.priors)
        print()

        occs = np.zeros((self.nb_classes, self.nb_words))

        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        print('Occurences:')
        print(occs)
        print()

        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.alpha
                down = np.sum(occs[c]) + self.nb_words * self.alpha
                self.like[c][w] = up / down
        print('Likelihoods:')
        print(self.like)
        print()

    def predict(self, bow):
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for w in range(self.nb_words):
                cnt = bow[w]
                prob += cnt * np.log(self.like[c][w])
            probs[c] = prob
        prediction = np.argmax(probs)
        return prediction

    def predict_multiply(self, bow):
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = self.priors[c]
            for w in range(self.nb_words):
                cnt = bow[w]
                prob *= self.like[c][w] ** cnt
            probs[c] = prob
        prediction = np.argmax(probs)
        return prediction


class_names = ["positive", "negative"]

model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), alpha=1)
model.fit(train_bow, pos_neg_values)

tp = 0
tn = 0
fp = 0
fn = 0

correct = 0

for i in range(len(test_bow)):
    prediction = model.predict_multiply(test_bow[i])
    prediction_class = class_names[prediction]
    if test_values[i] == 1:
        if prediction_class == "positive":
            tp += 1
            correct += 1
        else:
            fn += 1
    elif test_values[i] == 0:
        if prediction_class == "negative":
            tn += 1
            correct += 1
        else:
            fp += 1

print('accuracy: ')
print(correct / nb_test)
print()