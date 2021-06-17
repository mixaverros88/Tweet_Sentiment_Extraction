import gensim
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

    def vectorize_text(self):
        print(len(self.corpus))
        model = gensim.models.Word2Vec(self.corpus, min_count=1, window=5, sg=1)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.syn0)
        print('Word2Vec Vocabulary : ', len(model.wv.syn0))
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        # print('Word2Vec Vocabulary : ', model.wv.vocab)
        # print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))
        # print(vectors)
        # return vectors
        return model
