import pandas as pd
import numpy as np
import pickle

from helper_functions.clean_dataset.DataCleaning import DataCleaning


class ClassificationDto2:

    def __init__(self, cleaned_sample_data_frame, prob_array):
        self.cleaned_sample_data_frame = cleaned_sample_data_frame
        self.prob_array = prob_array

    # getter method
    def get_dataframe(self):
        return self.cleaned_sample_data_frame

    # getter method
    def get_array(self):
        return self.prob_array
