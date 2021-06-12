from sklearn import svm
import pickle


class SvmModel:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def results(self):
        model = svm.SVC(kernel='linear') # Linear Kernel
        model.fit(self.X_train, self.y_train)
        pickle.dump(model, open('serializedModels/svm.sav', 'wb'))
        return model.predict(self.X_test)
