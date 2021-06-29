from sklearn.linear_model import LogisticRegression
import pickle
import collections
from sklearn.model_selection import GridSearchCV
from helper.helper_functions.functions import get_models_best_parameters
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('Logistic Regression')
        # Grid Search
        # clf = LogisticRegression(solver='lbfgs', max_iter=10000)
        # param_grid = {'C': [120, 121, 122, 123, 124, 125, 126, 127, 128 ], 'penalty': ['l1', 'l2']}
        # model_gs = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
        # model_gs.fit(self.X_train, self.y_train)
        # get_models_best_parameters(model_gs, 'Logistic Regression')  # LogisticRegression(C=100)

        # https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
        # https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
        # define models and parameters
        model = LogisticRegression()
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [120, 121, 122, 123, 124, 125, 126, 127, 128 ]
        # define grid search
        grid = dict(solver=solvers, penalty=penalty, C=c_values)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(self.X_train, self.y_train)
        print(grid_result)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        get_models_best_parameters(grid_search, 'Logistic Regression')  # LogisticRegression(C=100)

        model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        y_score = model.fit(self.X_train, self.y_train).decision_function(self.X_test)
        pred = model.predict(self.X_test)
        Point = collections.namedtuple('Point', ['prediction', 'score'])
        p = Point(prediction=pred, score=y_score)
        return p
