from nltk import sent_tokenize, word_tokenize


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


def get_models_best_parameters(model, algo_name):
    print(algo_name + ' Best Parameters : ', model.best_estimator_)
