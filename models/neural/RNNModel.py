from keras.models import Sequential
from keras import layers


class RNNModel:

    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def results(self):
        print('RNN')
        embedding_dim = 50
        batch_size = 128

        model = Sequential()
        model.add(layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, input_length=self.max_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.X_train, self.Y_train, epochs=15, batch_size=batch_size, verbose=1)
        Y_pred = model.predict_classes(self.X_test, batch_size=batch_size)
        print(model.summary())
        return Y_pred
