import pandas as pd
import os


def read_train_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\train.csv', sep=',')


def read_sample_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\sample.csv', sep=',')


def read_test_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\test.csv', sep=',')
