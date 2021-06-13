from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions.tokenizer.functions import get_models_best_parameters


class KNeighborsModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        print('KNeighbors Classifier')
        # Grid Search
        # grid_params = {'n_neighbors': [3,5,11,19],'weights': ['uniform','distance'],'metric': ['euclidean', 'manhattan']}
        # kneighbors = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=5, n_jobs=1)
        # get_models_best_parameters(kneighbors, 'KNeighbors Classifier') # KNeighborsClassifier(metric='euclidean', weights='distance')

        model = KNeighborsClassifier(metric='euclidean', weights='distance')
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/kneighbors.sav', 'wb'))
        return model.predict(self.X_test)
