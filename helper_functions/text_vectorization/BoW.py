from sklearn.feature_extraction.text import CountVectorizer
import pickle


# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e

class BoW:

    def __init__(self, corpus, model_name):
        self.corpus = corpus
        self.model_name = model_name

    def vectorize_text(self):
        model = CountVectorizer()
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        # print(vectors)
        # print('BOW Vocabulary Size: ', len(vectorizer.vocabulary_))  # a list of unique words
        return vectors
