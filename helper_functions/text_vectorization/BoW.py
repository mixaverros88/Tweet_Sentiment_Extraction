from sklearn.feature_extraction.text import CountVectorizer
import pickle

# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e

class BoW:

    def __init__(self, corpus):
        self.corpus = corpus

    def vectorize_text(self):
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(self.corpus)
        pickle.dump(vectorizer, open('models/machine_learning/serialized/bag_of_words.sav', 'wb'))
        #print(vectors)
        #print('BOW Vocabulary Size: ', len(vectorizer.vocabulary_))  # a list of unique words
        return vectors
