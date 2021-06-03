from sklearn.feature_extraction.text import CountVectorizer


class BoW:

    def __init__(self, sentences):
        self.sentences = sentences
        self.vectorize_text()

    def vectorize_text(self):
        vectorizer = CountVectorizer()
        vectorizer.fit(self.sentences)
        print(vectorizer.vocabulary_)  # a list of unique words
        vector = vectorizer.transform(self.sentences)
        print(vector.toarray())