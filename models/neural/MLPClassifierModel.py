from sklearn.neural_network import MLPClassifier
import pickle
import collections
from sklearn.model_selection import GridSearchCV
from helper.helper_functions.functions import get_models_best_parameters


class MLPClassifierModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_name, *word2vec):
        if word2vec is None:
            self.X_train = X_train.todense()
            self.X_test = X_test.todense()
        else:
            self.X_train = X_train
            self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('MLPClassifier')
        # Grid Search
        # mlp = MLPClassifier(max_iter=100)
        # parameter_space = {'hidden_layer_sizes': [(5, 5, 5)], 'activation': ['tanh', 'relu'],
        #                    'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05],
        #                    'learning_rate': ['constant', 'adaptive'], }
        # model_gs = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        # model_gs.fit(self.X_train, self.y_train)
        # # MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),learning_rate='adaptive', max_iter=100)
        # get_models_best_parameters(model_gs, 'MLPClassifier')

        model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(5, 5, 5), learning_rate='adaptive',
                              max_iter=1000)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        pred = model.predict(self.X_test)
        # y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        y_score = None
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        p = Point(prediction=pred, score=y_score)
        return p
