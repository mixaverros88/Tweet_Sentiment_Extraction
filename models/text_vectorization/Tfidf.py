from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Tfidf:

    def __init__(self, corpus, model_name):
        self.corpus = corpus
        self.model_name = model_name

    def text_vectorization(self):
        model = TfidfVectorizer()
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        print('TfidfVectorizer get_feature_names(): ', model.get_feature_names())
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        print('Tfidf Vocabulary: ', model.vocabulary_)
        print(model)
        return vectors
