from nltk import sent_tokenize, word_tokenize
from collections import Counter


def tokenizing_sentences(data_frame):
    data_frame['tokenized_sents'] = data_frame.apply(lambda row: sent_tokenize(row['text']), axis=1)
    return data_frame


def tokenizing_sentences_and_words(data_frame):
    words = []
    for index, row in data_frame.iterrows():
        dd = []
        for word in word_tokenize(row['text']):
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
    print(list_word)
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
