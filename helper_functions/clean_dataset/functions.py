import re


# TODO: Removing stop words
def remove_stop_word():
    return


# TODO: Remove textID column from dataset

# TODO: Remove punctuation


def clean_html(text):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(cleaner, '', text)

# TODO: convert to lower case

# TODO: trim the text
