from sklearn.linear_model import LogisticRegression
import pickle


class LogisticRegressionModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        model = LogisticRegression(max_iter=13698)
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/logistic_regression.sav', 'wb'))
        return model.predict(self.X_test)
