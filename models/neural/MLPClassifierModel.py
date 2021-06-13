from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters


class MLPClassifierModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train.todense()
        self.X_test = X_test.todense()
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        print('MLPClassifier')
        # Grid Search
        mlp = MLPClassifier(max_iter=100)
        parameter_space = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)], 'activation': ['tanh', 'relu'],
                           'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05],
                           'learning_rate': ['constant', 'adaptive'], }
        model_gs = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        model_gs.fit(self.X_train, self.y_train)
        # MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),learning_rate='adaptive', max_iter=100)
        get_models_best_parameters(model_gs, 'MLPClassifier')

        model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(2, 2, 2), learning_rate='adaptive',
                              max_iter=100)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/MLPClassifierModel.sav', 'wb'))
        return model.predict(self.X_test)
