from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5

def logistic_regression_model_tuning(x_train, y_train):
    print('Logistic Regression Model Tuning')
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9]
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    grid_search(model, grid, x_train, y_train)


def mlp_classifier_model_tuning(x_train, y_train):
    print('MLPClassifier Model Tuning')
    model = MLPClassifier()
    grid = dict(hidden_layer_sizes=[(3, 3, 3), (5, 5, 5), (10, 10, 10)], activation=['tanh', 'relu'],
                solver=['sgd', 'adam'], alpha=[0.0001, 0.05, 1, 2],
                learning_rate=['constant', 'adaptive'], max_iter=[400])
    grid_search(model, grid, x_train, y_train)


def decision_tree_model_tuning(x_train, y_train):
    print('Decision Tree Model Tuning')
    model = DecisionTreeClassifier()
    grid = dict(max_leaf_nodes=list(range(2, 3, 5)), min_samples_split=[2, 3, 6],
                max_depth=np.arange(2, 3, 5))
    grid_search(model, grid, x_train, y_train)


def nb_model_tuning(x_train, y_train):
    print('Multinomial Naive Bayes Model Tuning')
    model = MultinomialNB()
    grid = dict(alpha=[1.0, 1.1, 1.5, 1.9, 2.0, 3.0, 4.0, 5.0])
    grid_search(model, grid, x_train, y_train)


def k_neighbors_model_tuning(x_train, y_train):
    print('K Neighbors Model Tuning')
    model = KNeighborsClassifier()
    grid = dict(n_neighbors=[3, 5, 11, 19, 25], weights=['uniform', 'distance'], metric=['euclidean', 'manhattan'])
    grid_search(model, grid, x_train, y_train)


def svm_model_tuning(x_train, y_train):
    print('Support Vector Machine Model Tuning')
    model = svm.SVC(kernel='linear', probability=True)
    grid = dict(C=np.linspace(start=1000, stop=10000, num=4, endpoint=True))
    grid_search(model, grid, x_train, y_train)


def grid_search(model, grid, x_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid.fit(x_train, y_train)
    print(grid_result)
    summarize_results(grid_result)


def summarize_results(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
