from nltk import sent_tokenize


def tokenizing_sentences(data_frame):
    data_frame['tokenized_sents'] = data_frame.apply(lambda row: sent_tokenize(row['text']), axis=1)
    return data_frame
