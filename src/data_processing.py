import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    return df


def summarize(df):

    print(df.info())

    print(df.describe())


def missing_values(df):

    return df.isnull().sum()


def transaction_statistics(df):

    return df[[
        "Amount",
        "Value"
    ]].describe()