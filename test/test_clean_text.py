from helper_functions.clean_dataset import DataCleaning as clean

text = 'Hello!!!, he said ---and went.   '
trimmed_text = clean.trim_text(text)
assert trimmed_text == 'Hello!!!, he said ---and went.'
assert clean.remove_punctuations_from_a_string(trimmed_text) == 'Hello he said and went'
assert clean.covert_text_to_lower_case(trimmed_text) == 'hello!!!, he said ---and went.'
