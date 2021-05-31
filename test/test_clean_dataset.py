from helper_functions.clean_dataset import DataCleaning as clean_dataset

text_with_html_tags = "<!DOCTYPE html><html><body><p><b>This text is bold</b></p></body></html>"

assert clean_dataset.clean_html(text_with_html_tags) == 'This text is bold'
