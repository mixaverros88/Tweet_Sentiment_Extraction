import pandas as pd


def read_test_data_set():
    return pd.read_csv('datasets/test.csv', sep=',')
