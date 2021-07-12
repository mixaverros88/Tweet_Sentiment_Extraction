from gensim.models import Word2Vec
import pickle
from utils.serializedModels import word2vec_over_sampling
from definitions import ROOT_DIR
from utils.functions import convert_corpus_to_vector_array_request


class Word2VecModel:

    def __init__(self, corpus, model_name=None):
        self.corpus = corpus
        if model_name is not None:
            self.model_name = model_name

    def text_vectorization(self):
        print(len(self.corpus))
        model = Word2Vec(sentences=self.corpus, window=5, min_count=1, workers=4)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.syn0)
        print('Word2Vec Vocabulary len : ', len(model.wv.syn0))
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        return model

    def text_vectorization_test_data_set(self):
        model = word2vec_over_sampling()
        vectors = convert_corpus_to_vector_array_request(model, self.corpus)
        print(model)
        return vectors
