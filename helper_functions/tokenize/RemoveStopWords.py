from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from autocorrect import Speller


class RemoveStopWords:
    stop_words = stopwords.words('english')
    word_tokens = []
    corrected_word_tokens = []
    sentences = []

    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.tokenize_text()
        self.spell_corrector()

    def tokenize_sentence(self):
        for index, row in self.data_frame.iterrows():
            self.sentences += [row['text']]
        return self.sentences

    def tokenize_text(self):
        for index, row in self.data_frame.iterrows():
            self.word_tokens += word_tokenize(row['text'])

    def spell_corrector(self):
        spell = Speller(lang='en')
        for word in self.word_tokens:
            corrected_word = spell(word)
            self.corrected_word_tokens += corrected_word
            if word != corrected_word:
                print(word + '   --- ' + spell(word))

    def remove_stop_words(self):
        # filtered_sentence = [w for w in self.word_tokens if not w.lower() in self.stop_words]
        filtered_sentence = []
        for word in self.word_tokens:
            if word not in self.stop_words:
                filtered_sentence.append(word)
        return filtered_sentence
