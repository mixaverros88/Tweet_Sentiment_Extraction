from sklearn import svm
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters


class SvmModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        print('Support Vector Machine SVM')
        # Grid Search
        # parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75), 'gamma': (1, 2, 3, 'auto'),
        #               'decision_function_shape': ('ovo', 'ovr'), 'shrinking': (True, False)}
        # model_gs = GridSearchCV(svm.SVC(), parameters, cv=5)
        # model_gs.fit(self.X_train, self.y_train)
        # # SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='linear')
        # get_models_best_parameters(model_gs, 'Support Vector Machine SVM')

        model = svm.SVC(kernel='linear')  # Linear Kernel
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/svm.sav', 'wb'))
        return model.predict(self.X_test)
