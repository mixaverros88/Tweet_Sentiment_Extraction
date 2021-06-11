from gensim.models import Word2Vec
import os


class Word2VecModel:

    def __init__(self, corpus):
        self.corpus = corpus

    def vectorize_text(self):
        model = Word2Vec(self.corpus, min_count=1, size=32)
        print(model)
        print('Word2Vec Vocabulary : ', model.wv.vocab)
        print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))
        # print(vectors)
        #return vectors
