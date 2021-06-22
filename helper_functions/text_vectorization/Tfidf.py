# https://stackoverflow.com/questions/45809560/remove-single-occurrences-of-words-in-vocabulary-tf-idf
# https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Tfidf:

    def __init__(self, corpus, model_name):
        self.corpus = corpus
        self.model_name = model_name

    def vectorize_text(self):
        # TODO: model tuning , fit vs fit_transform
        model = TfidfVectorizer(analyzer='word', stop_words='english')
        print(model)
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        print(model.get_feature_names())
        print(vectors)
        print('Tfidf Vocabulary Size: ', len(model.vocabulary_))
        return vectors
