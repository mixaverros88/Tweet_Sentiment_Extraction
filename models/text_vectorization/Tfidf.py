from sklearn.feature_extraction.text import TfidfVectorizer
from utils.serializedModels import tfidf_over_sampling, tfidf_under_sampling
import pickle
from definitions import ROOT_DIR


class Tfidf:

    def __init__(self, corpus, model_name=None):
        self.corpus = corpus
        if model_name is not None:
            self.model_name = model_name

    def text_vectorization(self):
        print('Tfidf')
        model = TfidfVectorizer()
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('TfidfVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary: ', model.vocabulary_)
        print(model)
        return vectors

    def text_vectorization_test_data_set_over_sampling(self):
        model = tfidf_over_sampling()  # Retrieve Model
        vectors = model.transform(self.corpus)
        print(model)
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('TfidfVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary : ', model.vocabulary_)
        return vectors

    def text_vectorization_test_data_set_under_sampling(self):
        model = tfidf_under_sampling()  # Retrieve Model
        vectors = model.transform(self.corpus)
        print(model)
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('TfidfVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary : ', model.vocabulary_)
        return vectors
