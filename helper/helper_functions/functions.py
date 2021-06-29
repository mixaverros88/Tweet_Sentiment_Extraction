from nltk import sent_tokenize, word_tokenize
from collections import Counter
import numpy as np


def tokenizing_sentences(data_frame):
    data_frame['tokenized_sents'] = data_frame.apply(lambda row: sent_tokenize(row['text']), axis=1)
    return data_frame


def tokenizing_sentences_and_words_data_frame(data_frame):
    words = []
    for index, row in data_frame.iterrows():
        dd = []
        for word in word_tokenize(row['text']):
            dd.append(word)
        words.append(dd)
    return words


def tokenizing_sentences_and_words_list(corpus):
    words = []
    for sentence in corpus:
        dd = []
        for word in word_tokenize(sentence):
            dd.append(word)
        words.append(dd)
    return words


def get_column_values_as_np_array(column_name, data_frame):
    return data_frame[column_name].values


def tokenize_sentence(data_frame):
    sentences = []
    for index, row in data_frame.iterrows():
        sentences += [row['text']]
    return sentences


def tokenize_text(data_frame):
    word_tokens = []
    for index, row in data_frame.iterrows():
        word_tokens += word_tokenize(row['text'])
    return word_tokens


def get_corpus(data_frame):
    sentences = ''
    for index, row in data_frame.iterrows():
        sentences += row['text']
    return [sentences]


def count_word_occurrences(data_frame, counter):
    """Returns a list of word that occurs base of give value """
    list_word = []
    words = tokenize_text(data_frame)
    counter_obj = Counter(words)
    # print(type(counter_obj.most_common()))
    for w in counter_obj.most_common():
        if w[1] <= counter:
            list_word.append(w[0])
            # print(w)
            # print(type(w[0]))
            # print(type(w[1]))
    print('list_word: ', list_word)
    print('Size of words with ' + str(counter) + ' or less occurrences ' + str(len(list_word)))
    return list_word


def remove_words_from_corpus(corpus, list_word):
    """Gets a list of sentences (corpus) and removes words from the given list (list_word)"""
    sentences = []
    for sentence in corpus:
        for word in list_word:
            sentence = sentence.replace(word, '')
        sentences.append(sentence)
    return sentences


def get_models_best_parameters(model, algo_name):
    print(algo_name + ' Best Parameters : ', model.best_estimator_)


def get_sentiment_as_array():
    return ['Negative', 'Neutral', 'Positive']


def sanitize_model_name(model_name):
    return model_name.lower().replace(" ", "_")


def convert_sentence_to_vector_array(word2vec_model, sentence):
    word_tokens = word_tokenize(sentence)
    np_vec = []
    for word in word_tokens:
        vec = word2vec_model.wv[str(word)]
        np_vec.append(vec)
    nn = np.array(np_vec)
    return np.average(np_vec, axis=0)


def convert_sentence_to_vector_array_request(word2vec_model, sentence):
    word_tokens = word_tokenize(sentence)
    np_vec = []
    for word in word_tokens:
        try:
            vec = word2vec_model.wv[str(word)]
        except:
            print(str(word) + ' is not in vocabulary word2vec')
        np_vec.append(vec)
    aa = np.average(np_vec, axis=0)
    nn = np.array(np_vec)
    return np_vec


def convert_data_frame_sentence_to_vector_array(word2vec_model, data_frame):
    x = data_frame['text'].apply(lambda sentence: convert_sentence_to_vector_array(word2vec_model, sentence))
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    x = np.concatenate(np.concatenate(x, axis=0), axis=0).reshape(-1, 100)
    return x
