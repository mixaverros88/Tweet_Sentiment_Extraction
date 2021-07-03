from sklearn.neural_network import MLPClassifier
import pickle
import collections
from helper.retrieve.serializedModels import bag_of_words_multi_layer_perceptron_classifier_over_sampling
from sklearn.model_selection import GridSearchCV
from helper.helper_functions.functions import get_models_best_parameters


def run_on_test_data_set(x, y):
    model = bag_of_words_multi_layer_perceptron_classifier_over_sampling()  # Retrieve Model
    return model.predict(x)


class MLPClassifierModel:

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
        print('MLPClassifier')
        # Grid Search
        # mlp = MLPClassifier(max_iter=100)
        # parameter_space = {'hidden_layer_sizes': [(5, 5, 5)], 'activation': ['tanh', 'relu'],
        #                    'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05],
        #                    'learning_rate': ['constant', 'adaptive'], }
        # model_gs = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        # model_gs.fit(self.X_train, self.y_train)
        # MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),learning_rate='adaptive', max_iter=100)
        # get_models_best_parameters(model_gs, 'MLPClassifier')

        model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(5, 5, 5), learning_rate='adaptive',
                              max_iter=1000)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        predictions = model.predict(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        return Point(prediction=predictions, score=None)
