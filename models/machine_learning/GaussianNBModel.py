from sklearn.naive_bayes import GaussianNB
import pickle


class GaussianNBModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train.todense()
        self.X_test = X_test.todense()
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/gaussian_nb.sav', 'wb'))
        return model.predict(self.X_test)
