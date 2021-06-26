from gensim.models import Word2Vec
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

from sklearn.linear_model import LogisticRegression
import numpy as np
import os


class Word2VecModel:

    def __init__(self, corpus, model_name):
        self.corpus = corpus
        self.model_name = model_name

    def text_vectorization(self):
        print(len(self.corpus))
        model = Word2Vec(sentences=self.corpus, window=5, min_count=1, workers=4)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.syn0)
        print('Word2Vec Vocabulary : ', len(model.wv.syn0))
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        # print('Word2Vec Vocabulary : ', model.wv.vocab)
        # print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))
        # print(vectors)
        # return vectors
        return model
