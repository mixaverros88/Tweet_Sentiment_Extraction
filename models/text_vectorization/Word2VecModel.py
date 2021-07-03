from gensim.models import Word2Vec
import pickle
from helper.retrieve.serializedModels import word2vec_over_sampling
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

from sklearn.linear_model import LogisticRegression
import numpy as np
import os


class Word2VecModel:

    def __init__(self, corpus, *model_name):
        self.corpus = corpus
        self.model_name = str(model_name)

    def text_vectorization(self):
        print(len(self.corpus))
        model = Word2Vec(sentences=self.corpus, window=5, min_count=1, workers=4)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.syn0)
        print('Word2Vec Vocabulary len : ', len(model.wv.syn0))
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        # print('Word2Vec Vocabulary : ', model.wv.vocab)
        # print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))
        # print(vectors)
        # return vectors
        return model

    def text_vectorization_test_data_set(self):
        model = word2vec_over_sampling()  # Retrieve Model
        vectors = model.transform(self.corpus)
        print(model)
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('TfidfVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary : ', model.vocabulary_)
        return vectors