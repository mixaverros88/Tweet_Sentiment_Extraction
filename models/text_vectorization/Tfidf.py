from sklearn.feature_extraction.text import TfidfVectorizer
from utils.serializedModels import tfidf_over_sampling, tfidf_under_sampling
from utils.functions import compute_elapsed_time, save_models
import time

""""TF-IDF"""


class Tfidf:

    def __init__(self, corpus, model_name=None):
        self.corpus = corpus
        if model_name is not None:
            self.model_name = model_name

    def text_vectorization(self):
        print('Tfidf')
        start_time = time.time()
        model = TfidfVectorizer()
        vectors = model.fit_transform(self.corpus)
        save_models(model, self.model_name)
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('TfidfVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary: ', model.vocabulary_)
        print(model)
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return vectors

    def text_vectorization_test_data_set_over_sampling(self):
        model = tfidf_over_sampling()
        return model.transform(self.corpus)

    def text_vectorization_test_data_set_under_sampling(self):
        model = tfidf_under_sampling()
        return model.transform(self.corpus)
