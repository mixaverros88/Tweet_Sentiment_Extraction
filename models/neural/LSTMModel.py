import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import re
# https://github.com/sanjay-raghu/sentiment-analysis-using-LSTM-keras/blob/master/lstm-sentiment-analysis-data-imbalance-keras.ipynb

class LSTMModel:

    def __init__(self,X, Y,X_train, X_test, y_train, y_test, model_name):
        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name

    def results(self):
        print('LSTM')
        max_features = 2000
        embed_dim = 128
        lstm_out = 196
        batch_size = 128

        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=self.X.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=42)
        # model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test ), epochs=20, batch_size=64, shuffle=True)
        model.fit(X_train, Y_train, epochs=15, batch_size=batch_size, verbose=1)
        Y_pred = model.predict_classes(self.X_test, batch_size=batch_size)
        print('Y_pred', Y_pred)
        return Y_pred
