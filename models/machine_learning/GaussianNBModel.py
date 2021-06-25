from sklearn.naive_bayes import MultinomialNB
import pickle
import collections
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters
import numpy as np


# https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
# https://dzone.com/articles/scikit-learn-using-gridsearch-to-tune-the-hyperpar

class GaussianNBModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, *word2vec):
        if word2vec is None:
            self.X_train = x_train.todense()
            self.X_test = x_test.todense()
        else:
            self.X_train = x_train
            self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        # TODO: gaussian vs multinomial naive bayes
        print('Multinomial Naive Bayes')
        # Grid Search
        # params = {'alpha': [1.0, 1.1, 1.5, 1.9, 2.0, 3.0, 4.0, 5.0], }
        # model_gs = GridSearchCV(MultinomialNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
        # print(model_gs)
        # model_gs.fit(self.X_train, self.y_train)
        # get_models_best_parameters(model_gs, 'Multinomial Naive Bayes')

        model = MultinomialNB(alpha=1.5)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        pred = model.predict(self.X_test)
        # y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        y_score = None
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        p = Point(prediction=pred, score=y_score)
        return p
