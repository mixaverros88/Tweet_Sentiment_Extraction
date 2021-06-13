from keras.models import Sequential
from keras import layers


class RNN:

    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def results(self):
        embedding_dim = 50

        model = Sequential()
        model.add(layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, input_length=self.max_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
