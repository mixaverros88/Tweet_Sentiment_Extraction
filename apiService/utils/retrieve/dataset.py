import pandas as pd


def read_test_data_set():
    """Retrieve Test Data Set"""
    return pd.read_csv('datasets/test.csv', sep=',')
