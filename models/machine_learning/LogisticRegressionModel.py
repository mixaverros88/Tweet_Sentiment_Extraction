from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters


class LogisticRegressionModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        print('Logistic Regression')
        # Grid Search
        # clf = LogisticRegression()
        # param_grid = {'C': [0.01, 0.1, 1, 2, 10, 100], 'penalty': ['l1', 'l2']}
        # model_gs = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
        # model_gs.fit(self.X_train, self.y_train)
        # get_models_best_parameters(model_gs, 'Logistic Regression')  # LogisticRegression(C=100)

        model = LogisticRegression(C=0.1)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/logistic_regression.sav', 'wb'))
        return model.predict(self.X_test)
