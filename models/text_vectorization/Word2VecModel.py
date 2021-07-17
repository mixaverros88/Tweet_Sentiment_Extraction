from gensim.models import Word2Vec
from utils.functions import compute_elapsed_time, save_models
from utils.serializedModels import word2vec_over_sampling, word2vec_under_sampling
import time
from utils.functions import convert_corpus_to_vector_array_request

""""Word2Vec"""


class Word2VecModel:

    def __init__(self, corpus, model_name=None):
        self.corpus = corpus
        if model_name is not None:
            self.model_name = model_name

    def text_vectorization(self):
        start_time = time.time()
        print(len(self.corpus))
        model = Word2Vec(sentences=self.corpus, window=5, min_count=1, workers=4)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.syn0)
        print('Word2Vec Vocabulary len : ', len(model.wv.syn0))
        save_models(model, self.model_name)
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return model

    def text_vectorization_test_data_set_over_sampling(self):
        model = word2vec_over_sampling()
        return convert_corpus_to_vector_array_request(model, self.corpus)

    def text_vectorization_test_data_set_under_sampling(self):
        model = word2vec_under_sampling()
        return convert_corpus_to_vector_array_request(model, self.corpus)
