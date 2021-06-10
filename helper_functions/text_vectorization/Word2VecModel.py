from gensim.models import Word2Vec
import os


class Word2VecModel:

    def __init__(self, corpus):
        self.corpus = corpus

    def vectorize_text(self):
        print('Corpus: ', self.corpus)
        model = Word2Vec(self.corpus, min_count=1, size=32)
        # model.build_vocab(self.corpus, progress_per=1000)
        model.train(self.corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(os.path.abspath(os.curdir) + '\\helper_functions\\text_vectorization\\saved\\myModel.model')
        print('Word2Vec Vocabulary : ', model.wv.vocab)
        print('Word2Vec Vocabulary Size: ', len(model.wv.vocab))
        print(model.wv.most_similar("what"))
        # print(model.wv.similarity(w1="great", w2="awesome"))

    # TODO: plot
