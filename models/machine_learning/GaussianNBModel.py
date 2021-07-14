from sklearn.naive_bayes import MultinomialNB
import pickle
import collections
from utils.serializedModels import bag_of_words_nb_over_sampling, bag_of_words_nb_under_sampling
from definitions import ROOT_DIR


def run_on_test_data_set_over_sampling(x):
    model = bag_of_words_nb_over_sampling()
    return model.predict(x)


def run_on_test_data_set_under_sampling(x):
    model = bag_of_words_nb_under_sampling()
    return model.predict(x)


# https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
# https://dzone.com/articles/scikit-learn-using-gridsearch-to-tune-the-hyperpar

class GaussianNBModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space, *word2vec):
        if word2vec is None:
            self.X_train = x_train.todense()
            self.X_test = x_test.todense()
        else:
            self.X_train = x_train
            self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Multinomial Naive Bayes')
        # nb_model_tuning(self.x_train, self.y_train)
        model = MultinomialNB(alpha=1.5)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
