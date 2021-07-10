from nltk import sent_tokenize, word_tokenize
from collections import Counter
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd


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


def convert_numpy_array_to_array_of_arrays(numpy_array):
    return_array = []
    for i in numpy_array:
        dd = []
        for word in word_tokenize(i):
            dd.append(word)
        return_array.append(dd)
    return return_array


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
        word_tokens = word_tokenize(sentence)
        for word in list_word:
            if word in word_tokens:
                # sentence = re.sub(r'\b' + word + '\\b', '', sentence)
                word_tokens.remove(word)
                # print(word)
                # print(sentence)
                # sentence = sentence.replace(word, '')
            # print(sentence)
        sentence = TreebankWordDetokenizer().detokenize(word_tokens)
        # sentence = re.sub('\s+', ' ', sentence)  # Remove Double Spaces
        # print(sentence)
        sentences.append(sentence)
    return sentences


def get_models_best_parameters(model, algo_name):
    print(algo_name + ' Best Estimator : ', model.best_estimator_)
    print(algo_name + ' Best Parameters : ', model.best_params_)


def get_sentiment_as_array():
    return ['Negative', 'Neutral', 'Positive']


def sanitize_model_name(model_name):
    return model_name.lower().replace(" ", "_")


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

def get_indices(df,col,n):
    """
    Get the indices of dataframe where exist more than n tokens in a specific column

    Parameters:

       df(pandas dataframe)
       n(int): threshold value for minimum words
       col(string): column name

    """


    tmp = []
    for i in range(len(df)):#df.iterrows() wasnt working for me
        if len(word_tokenize(df[col][i])) < n:
            tmp.append(i)
    return tmp

def convert_sentence_to_vector_array2(word2vec_model, sentences):
    data_frame = pd.DataFrame({'text': sentences})
    #tmp = get_indices(data_frame, 'text', 2)
    #data_frame = data_frame.drop(tmp)
    x = data_frame['text'].apply(lambda sentence: convert_sentence_to_vector_array(word2vec_model, sentence))
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    x = np.concatenate(np.concatenate(x, axis=0), axis=0).reshape(-1, 100)
    return x


def convert_sentence_to_vector_array_request(word2vec_model, sentence):
    word_tokens = word_tokenize(sentence)
    np_vec = []
    for word in word_tokens:
        try:
            vec = word2vec_model.wv[str(word)]
            np_vec.append(vec)
        except:
            print(str(word) + ' is not in vocabulary word2vec')
    return np_vec


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


def convert_data_frame_sentence_to_vector_array(word2vec_model, data_frame):
    x = data_frame['text'].apply(lambda sentence: convert_sentence_to_vector_array(word2vec_model, sentence))
    x = x.to_numpy()
    x = x.reshape(-1, 1)
    x = np.concatenate(np.concatenate(x, axis=0), axis=0).reshape(-1, 100)
    return x


def count_words_per_sentence(sentence):
    sentence = str(sentence)
    return len(sentence.split())


def count_the_most_common_words_in_data_set(data_set, column, counter):
    # get most common words in given dataset
    all_words = []
    for line in list(data_set[column]):
        line = str(line)
        words = line.split()
        for word in words:
            all_words.append(word.lower())

    return Counter(all_words).most_common(counter)


def count_the_most_common_words_in_data_set_convert(list):
    res_list = [x[0] for x in list]
    print(res_list)
    return res_list


def map_sentiment(sentiment):
    if sentiment == 0:
        return 'Negative'
    if sentiment == 1:
        return 'Neutral'
    if sentiment == 2:
        return 'Positive'


def convert_text_to_data_frame_of_one_row(text):
    request_text = {'text': [text]}
    return pd.DataFrame(request_text)


def convert_list_to_numpy_array(vec_list):
    return np.asarray(vec_list)
