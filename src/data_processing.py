import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer


def create_aggregate_features(df):

    customer_features = df.groupby(
        "CustomerId"
    ).agg({

        "Amount":[
            "sum",
            "mean",
            "count",
            "std"
        ]

    })

    customer_features.columns=[

        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount"

    ]

    customer_features.reset_index(
        inplace=True
    )

    return customer_features


def extract_time_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["hour"]=df[
        "TransactionStartTime"
    ].dt.hour

    df["day"]=df[
        "TransactionStartTime"
    ].dt.day

    df["month"]=df[
        "TransactionStartTime"
    ].dt.month

    df["year"]=df[
        "TransactionStartTime"
    ].dt.year

    return df


def build_pipeline(df):

    numeric = df.select_dtypes(
        include=np.number
    ).columns

    categorical = df.select_dtypes(
        exclude=np.number
    ).columns

    num_pipe = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    cat_pipe = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ])

    preprocessor=ColumnTransformer([

        (
            "num",
            num_pipe,
            numeric
        ),

        (
            "cat",
            cat_pipe,
            categorical
        )

    ])

    return preprocessor