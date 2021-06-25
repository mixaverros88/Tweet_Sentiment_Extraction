from sklearn.feature_extraction.text import CountVectorizer
import pickle

# https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e
# https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
"""We call vectorization the general process of turning a collection of text documents into numerical feature vectors. 
This specific strategy (tokenization, counting and normalization) is called the Bag of Words or "Bag of n-grams" 
representation. Documents are described by word occurrences while completely ignoring the relative position
 information of the words in the document."""


class BoW:

    def __init__(self, corpus, model_name):
        self.corpus = corpus
        self.model_name = model_name

    def text_vectorization(self):
        # TODO: model tuning , fit vs fit_transform
        model = CountVectorizer()
        print(model)
        vectors = model.fit_transform(self.corpus)
        pickle.dump(model, open('serializedModels/' + self.model_name + '.sav', 'wb'))
        print('CountVectorizer get_feature_names(): ', model.get_feature_names())
        # print(vectors)
        print('BOW Vocabulary Size: ', len(model.vocabulary_))
        return vectors
