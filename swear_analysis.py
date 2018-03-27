# Author Ben Newman, much code taken from 
# How to make a racist AI without really trying 
# https://gist.github.com/rspeer/ef750e7e407e04894cb3b78a82d66aed

import numpy as np
import pandas as pd
import gensim
from gensim.models.keyedvectors import KeyedVectors


DATA_PATH = "../data/"
# models are stored in gensim format for easier access later:
# https://radimrehurek.com/gensim/models/word2vec.html
MODEL_PATH = "models/"


def load_embeddings(filename):
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText, and ConceptNet Numberbatch. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)

    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')

def print_closest_words(vectors, targets):
    for t in targets:
        print("'%s':"%t)
        closest = vectors.most_similar(t)
        for c in closest:
            print("\t%s : %f" %(c[0], c[1]))

#def load_gensim_embeddings(filename)
words_of_interest = ["fuck", "fucking", "friggen", "shit", "damn", "poop", "darn", "frog"]

#word_vectors = KeyedVectors.load_word2vec_format('%sglove.42B.300d.txt'%DATA_PATH, binary=False)
#print("\n\n------GLOVE 42B 300D------\n")
#print_closest_words(word_vectors, words_of_interest)
#word_vectors.save('%sgolve.42B.300d.model'%MODEL_PATH)
#
#word_vectors = KeyedVectors.load_word2vec_format('%sglove.twitter.27B.25d.txt'%DATA_PATH, binary=False)
#print("\n\n------GLOVE TWITTER 27B 25D------\n")
#print_closest_words(word_vectors, words_of_interest)
#word_vectors.save('%sgolve.twitter.27B.25d.model'%MODEL_PATH)

word_vectors = KeyedVectors.load_word2vec_format('%sglove.twitter.27B.50d.txt'%DATA_PATH, binary=False)
print("\n\n------GLOVE TWITTER 27B 50D------\n")
print_closest_words(word_vectors, words_of_interest)
word_vectors.save('%sgolve.twitter.27B.50d.model'%MODEL_PATH)

word_vectors = KeyedVectors.load_word2vec_format('%sglove.twitter.27B.100d.txt'%DATA_PATH, binary=False)
print("\n\n------GLOVE TWITTER 27B 100D------\n")
print_closest_words(word_vectors, words_of_interest)
word_vectors.save('%sgolve.twitter.27B.100d.model'%MODEL_PATH)

word_vectors = KeyedVectors.load_word2vec_format('%sglove.twitter.27B.200d.txt'%DATA_PATH, binary=False)
print("\n\n------GLOVE TWITTER 27B 200D------\n")
print_closest_words(word_vectors, words_of_interest)
word_vectors.save('%sgolve.twitter.27B.200d.model'%MODEL_PATH)
