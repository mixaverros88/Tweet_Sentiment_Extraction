from nltk import word_tokenize
from collections import Counter
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
from definitions import ROOT_DIR


def tokenizing_dataframe(data_frame):
    """" Returns every row as a list of tokens.
    [['i', 'have', 'have', 'respond', 'if', 'i', 'be', 'go'], ['soo', 'sad', 'i', 'will', 'miss', 'you', 'here', 'in']]
    """
    sentence = []
    for index, row in data_frame.iterrows():
        words = []
        for word in word_tokenize(row['text']):
            words.append(word)
        sentence.append(words)
    return sentence


def get_column_values_as_np_array(column_name, data_frame):
    return data_frame[column_name].values


def convert_data_frame_to_list(data_frame):
    """ e.g.['last of day', 'also really excite good tweet in']"""
    sentences = []
    for index, row in data_frame.iterrows():
        sentences += [row['text']]
    return sentences


def tokenize_text(data_frame):
    word_tokens = []
    for index, row in data_frame.iterrows():
        word_tokens += word_tokenize(row['text'])
    return word_tokens


def count_word_occurrences(data_frame, counter):
    """Returns a list of word that occurs the most base of given value """
    list_word = []
    words = tokenize_text(data_frame)
    counter_obj = Counter(words)
    for w in counter_obj.most_common():
        if w[1] <= counter:
            list_word.append(w[0])
    print('list_word: ', list_word)
    print('Size of words with ' + str(counter) + ' or less occurrences ' + str(len(list_word)))
    return list_word


def remove_words_from_corpus(corpus, list_word):
    """Gets a list of sentences (corpus) and removes words from the given list (list_word)"""
    sentences = []
    for sentence in corpus:
        word_tokens = word_tokenize(sentence)
        for word in list_word:
            if word in word_tokens:
                word_tokens.remove(word)
        sentence = TreebankWordDetokenizer().detokenize(word_tokens)
        sentences.append(sentence)
    return sentences


def get_sentiment_as_array():
    return ['Negative', 'Neutral', 'Positive']


def sanitize_model_name(model_name):
    return model_name.lower().replace(" ", "_")


def convert_list_of_tuples_to_list(most_common_words_tuple):
    """e.g.
    [('i', 2198), ('be', 1836), ('to', 1300), ('the', 1174), ('have', 966), ('you', 906), ('a', 833), ('my', 745)]
    to
    ['i', 'be', 'to', 'the', 'have', 'you', 'a','my']
    """
    most_common_words = [x[0] for x in most_common_words_tuple]
    return most_common_words


def count_the_most_common_words_in_data_set(data_set, column, counter):
    """get most common words of given dataset"""
    all_words = []
    for line in list(data_set[column]):
        line = str(line)
        words = line.split()
        for word in words:
            all_words.append(word.lower())

    return Counter(all_words).most_common(counter)


def count_words_per_sentence(sentence):
    sentence = str(sentence)
    return len(sentence.split())


def convert_corpus_to_vector_array_request(word2vec_model, corpus):
    np_vec = []
    for sentence in corpus:
        word_tokens = word_tokenize(str(sentence))
        for word in word_tokens:
            try:
                vec = word2vec_model.wv[str(word)]
                np_vec.append(vec)
            except:
                print(str(word) + ' is not in vocabulary word2vec')
    return np_vec


def convert_sentence_to_vector_array(word2vec_model, sentence):
    word_tokens = word_tokenize(sentence)
    np_vec = []
    for word in word_tokens:
        try:
            vec = word2vec_model.wv[str(word)]
            np_vec.append(vec)
        except:
            print(str(word) + ' is not in vocabulary word2vec')
    return np.average(np_vec, axis=0)


def convert_data_frame_sentence_to_vector_array(word2vec_model, data_frame):
    x = data_frame['text'].apply(lambda sentence: convert_sentence_to_vector_array(word2vec_model, sentence))
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    x = np.concatenate(np.concatenate(x, axis=0), axis=0).reshape(-1, 100)
    return x


intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),  # 60 * 60 * 24
    ('hours', 3600),  # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
)


def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


def compute_elapsed_time(start_time, end_time, model_name):
    elapsed_time = end_time - start_time
    print(model_name + ' Elapsed Time : ' + str(display_time(elapsed_time)))


def save_models(model, model_name):
    pickle.dump(model, open(ROOT_DIR + '/apiService/serializedModels/' + model_name + '.sav', 'wb'))
