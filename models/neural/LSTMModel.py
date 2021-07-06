from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
import tensorflow.python.keras.backend as K

sess = K.get_session()

# https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras
# https://github.com/sanjay-raghu/sentiment-analysis-using-LSTM-keras/blob/master/lstm-sentiment-analysis-data-imbalance-keras.ipynb
# https://github.com/nagypeterjob/Sentiment-Analysis-NLTK-ML-LSTM/blob/master/lstm.ipynb

class LSTMModel:

    def __init__(self, x_train, x_test, y_train, y_test, model_name, size):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.size = size

    def results(self):
        print('LSTM')
        max_features = 2000
        embed_dim = 128
        lstm_out = 196
        batch_size = 128

        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=self.size))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(self.x_train, self.y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)
        Y_pred = model.predict_classes(self.x_test, batch_size=batch_size)
        print('Y_pred', Y_pred)
        return Y_pred
