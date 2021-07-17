import pandas as pd
from definitions import ROOT_DIR

print(__name__)
"""Retrieve Data set"""


def read_train_data_set():
    return pd.read_csv(ROOT_DIR + '/datasets/initial/train.csv', sep=',')


def read_test_data_set():
    return pd.read_csv(ROOT_DIR + '/datasets/initial/test.csv', sep=',')


def read_cleaned_train_data_set_under_sampling():
    return pd.read_csv(ROOT_DIR + '/datasets/cleaned/Train_DatasetUnder_Sampling_dataframe_cleaned.csv', sep=',')


def read_cleaned_train_data_set_over_sampling():
    return pd.read_csv(ROOT_DIR + '/datasets/cleaned/Train_DatasetOver_Sampling_dataframe_cleaned.csv', sep=',')


def read_cleaned_test_data_set():
    return pd.read_csv(ROOT_DIR + '/datasets/cleaned/Test_Dataset_dataframe_cleaned.csv', sep=',')
