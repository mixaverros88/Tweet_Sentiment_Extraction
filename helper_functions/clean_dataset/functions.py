import re


# TODO: remove empty text

# TODO: Removing stop words
def remove_stop_word():
    return


def remove_column_from_data_frame(data_frame, column_name):
    """Remove a column from give data frame """
    data_frame.drop(column_name, axis=1, inplace=True)


def sanitize_data_frame(data_frame):
    for index, row in data_frame.iterrows():
        print(row['text'])
        text = trim_text(covert_text_to_lower_case(clean_html(remove_punctuations_from_a_string(row['text']))))
        data_frame.loc[index, 'text'] = text

def remove_punctuations_from_a_string(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    text_without_punctuations = ''
    for char in text:
        if char not in punctuations:
            text_without_punctuations = text_without_punctuations + char
    return text_without_punctuations


def clean_html(text):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(cleaner, '', text)


def covert_text_to_lower_case(text):
    return text.lower()


def trim_text(text):
    return text.strip()
