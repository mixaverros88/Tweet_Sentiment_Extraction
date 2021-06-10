from sklearn.feature_extraction.text import CountVectorizer

class BoW:

    def __init__(self, data_frame, corpus):
        self.data_frame = data_frame
        self.corpus = corpus

    def vectorize_text(self):
        arr = []
        vectorizer = CountVectorizer()
        vectorizer.fit(self.corpus)
        print('Vocabulary: ', vectorizer.vocabulary_)  # a list of unique words
        print('Vocabulary Size: ', len(vectorizer.vocabulary_))  # a list of unique words
        # freqs = zip(vectorizer.get_feature_names(), vectorizer.sum(axis=0))
        # # sort from largest to smallest
        # print(sorted(freqs, key=lambda x: -x[1]))

        for index, row in self.data_frame.iterrows():
            vector = vectorizer.transform(row['tokenized_sents'])
            arr.append(vector.toarray())
            arr.append(row['sentiment'])
            print('vector.toarray()', vector.toarray())
        return arr
