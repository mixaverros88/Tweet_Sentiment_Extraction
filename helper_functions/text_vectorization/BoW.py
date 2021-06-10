from sklearn.feature_extraction.text import CountVectorizer


# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e
class BoW:

    def __init__(self, data_frame, corpus):
        self.data_frame = data_frame
        self.corpus = corpus

    def vectorize_text(self):
        print(self.corpus)
        arr = []
        vectorizer = CountVectorizer()
        vectorizer.fit(self.corpus)
        # Now, we can inspect how our vectorizer vectorized the text
        # This will print out a list of words used, and their index in the vectors
        print('BOW Vocabulary : ', vectorizer.vocabulary_)  # a list of unique words
        print('BOW Vocabulary Size: ', len(vectorizer.vocabulary_))  # a list of unique words

        for index, row in self.data_frame.iterrows():
            vector = vectorizer.transform(row['tokenized_sents'])
            arr.append(vector.toarray())
            arr.append(row['sentiment'])
            print('vector.toarray()', vector.toarray())
        return arr
