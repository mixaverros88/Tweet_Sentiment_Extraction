from gensim.models import Word2Vec


class Word2Vec:

    def __init__(self, corpus):
        self.corpus = corpus

    def vectorize_text(self):
        model = Word2Vec(self.corpus, min_count=1, size=32)
        model.save('/helper_functions/text_vectorization/saved/word2vec.model')
        print(model.most_similar('man'))
        print('Word2Vec Vocabulary : ', model.wv.vocab)
        print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))

    # TODO: plot
