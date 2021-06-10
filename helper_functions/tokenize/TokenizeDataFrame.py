from nltk.tokenize import word_tokenize


class TokenizeDataFrame:
    word_tokens = []
    sentences = []

    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.tokenize_text()

    def tokenize_sentence(self):
        for index, row in self.data_frame.iterrows():
            self.sentences += [row['text']]
        return self.sentences

    def tokenize_text(self):
        for index, row in self.data_frame.iterrows():
            self.word_tokens += word_tokenize(row['text'])
