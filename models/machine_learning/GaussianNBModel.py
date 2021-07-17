from sklearn.naive_bayes import MultinomialNB
import collections
from utils.model_tuning import nb_model_tuning
import time

from utils.functions import compute_elapsed_time, save_models

# https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
# https://dzone.com/articles/scikit-learn-using-gridsearch-to-tune-the-hyperpar


class GaussianNBModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, param_space, *word2vec):
        if word2vec is None:
            self.x_train = x_train.todense()
            self.x_test = x_test.todense()
        else:
            self.x_train = x_train
            self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.param_space = param_space

    def results(self):
        print('Multinomial Naive Bayes')
        start_time = time.time()
        # nb_model_tuning(self.x_train, self.y_train)
        model = MultinomialNB(alpha=self.param_space.get('alpha'))
        model.fit(self.x_train, self.y_train)
        score = model.predict_proba(self.x_test)
        save_models(model, self.model_name)
        predictions = model.predict(self.x_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        end_time = time.time()
        compute_elapsed_time(start_time, end_time, self.model_name)
        return Point(prediction=predictions, score=score)
