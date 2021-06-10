import pandas as pd
import os


def read_train_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\initial\\train.csv', sep=',')


def read_sample_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\initial\\sample.csv', sep=',')


def read_test_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\initial\\test.csv', sep=',')


def read_cleaned_train_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Train_Dataset_dataframe_cleaned.csv',
                       sep=',')


def read_cleaned_sample_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Sample_Dataset_dataframe_cleaned.csv',
                       sep=',')


def read_cleaned_test_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\cleaned\\Test_Dataset_dataframe_cleaned.csv',
                       sep=',')
