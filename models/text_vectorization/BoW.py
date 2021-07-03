from sklearn.feature_extraction.text import CountVectorizer
import pickle
from helper.retrieve.serializedModels import bag_of_words_over_sampling


class BoW:

    def __init__(self, corpus, *model_name):
        self.corpus = corpus
        self.model_name = str(model_name)

    def text_vectorization(self):
        model = CountVectorizer()
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        print(model)
        print('CountVectorizer get_feature_names(): ', model.get_feature_names())
        print('CountVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('BOW Vocabulary Size: ', len(model.vocabulary_))
        print('BOW Vocabulary : ', model.vocabulary_)
        return vectors

    def text_vectorization_test_data_set(self):
        model = bag_of_words_over_sampling()  # Retrieve Model
        vectors = model.transform(self.corpus)
        print(model)
        print('CountVectorizer get_feature_names(): ', model.get_feature_names())
        print('CountVectorizer len get_feature_names(): ', len(model.get_feature_names()))
        print('BOW Vocabulary Size: ', len(model.vocabulary_))
        print('BOW Vocabulary : ', model.vocabulary_)
        return vectors
