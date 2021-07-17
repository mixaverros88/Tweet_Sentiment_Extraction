from sklearn.feature_extraction.text import CountVectorizer
import time
from utils.serializedModels import bag_of_words_over_sampling, bag_of_words_under_sampling
from utils.functions import compute_elapsed_time, save_models

""""Bag Of Words"""


class BoW:

    def __init__(self, corpus, model_name=None):
        self.corpus = corpus
        if model_name is not None:
            self.model_name = str(model_name)

    def text_vectorization(self):
        print('BOW Vocabulary')
        model = CountVectorizer()
        start_time = time.time()
        vectors = model.fit_transform(self.corpus)
        save_models(model, self.model_name)
        print(model)
        print('CountVectorizer get_feature_names(): ', model.get_feature_names())
        print('CountVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('BOW Vocabulary Size: ', len(model.vocabulary_))
        print('BOW Vocabulary : ', model.vocabulary_)
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return vectors

    def text_vectorization_test_data_set_over_sampling(self):
        model = bag_of_words_over_sampling()  # Retrieve Model
        return model.transform(self.corpus)

    def text_vectorization_test_data_set_under_sampling(self):
        model = bag_of_words_under_sampling()  # Retrieve Model
        return model.transform(self.corpus)
