from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class RemoveStopWords:
    stop_words = stopwords.words('english')
    word_tokens = []

    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.tokenize_text()

    def tokenize_text(self):
        for index, row in self.data_frame.iterrows():
            self.word_tokens += word_tokenize(row['text'])

    def remove_stop_words(self):
        # filtered_sentence = [w for w in self.word_tokens if not w.lower() in self.stop_words]
        filtered_sentence = []
        for word in self.word_tokens:
            if word not in self.stop_words:
                filtered_sentence.append(word)
        return filtered_sentence
