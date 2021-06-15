import pandas as pd
import os
from pathlib import Path

path = Path()


def read_train_data_set():
    print(path.parent.absolute().parent)
    return pd.read_csv(os.path.abspath(path.parent.absolute().parent) + '\\datasets\\initial\\train.csv', sep=',')


def read_test_data_set():
    return pd.read_csv(os.path.abspath(path.parent.absolute().parent) + '\\datasets\\initial\\test.csv', sep=',')


def read_cleaned_train_data_set_under_sampling():
    return pd.read_csv(
        os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Train_DatasetUnder_Sampling_dataframe_cleaned.csv',
        sep=',')


def read_cleaned_train_data_set_over_sampling():
    return pd.read_csv(
        os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Train_DatasetOver_Sampling_dataframe_cleaned.csv',
        sep=',')


def read_cleaned_test_data_set():
    return pd.read_csv(
        os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Test_Dataset_dataframe_cleaned.csv',
        sep=',')
