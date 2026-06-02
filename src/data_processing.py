import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans


def extract_time_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = \
        df["TransactionStartTime"].dt.hour

    df["transaction_day"] = \
        df["TransactionStartTime"].dt.day

    df["transaction_month"] = \
        df["TransactionStartTime"].dt.month

    df["transaction_year"] = \
        df["TransactionStartTime"].dt.year

    return df


def create_aggregate_features(df):

    agg = df.groupby(
        "CustomerId"
    ).agg({

        "Amount":[
            "sum",
            "mean",
            "count",
            "std"
        ]

    })

    agg.columns = [

        "total_transaction_amount",

        "average_transaction_amount",

        "transaction_count",

        "transaction_std"

    ]

    agg = agg.reset_index()

    return agg


def create_rfm(df):

    snapshot_date = \
        df["TransactionStartTime"].max()

    rfm = df.groupby(
        "CustomerId"
    ).agg({

        "TransactionStartTime":
        lambda x:
        (
            snapshot_date
            -
            x.max()
        ).days,

        "TransactionId":"count",

        "Amount":"sum"

    })

    rfm.columns = [

        "Recency",

        "Frequency",

        "Monetary"

    ]

    return rfm


def assign_risk_label(rfm):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        rfm
    )

    kmeans = KMeans(

        n_clusters=3,

        random_state=42

    )

    rfm["cluster"] = \
        kmeans.fit_predict(
            scaled
        )

    risk_cluster = \

        rfm.groupby(
            "cluster"
        )["Frequency"]\
        .mean()\
        .idxmin()

    rfm["is_high_risk"] = (

        rfm["cluster"]
        ==
        risk_cluster

    ).astype(int)

    return rfm


def build_pipeline(df):

    numeric_cols = \
        df.select_dtypes(
            include=np.number
        ).columns

    categorical_cols = \
        df.select_dtypes(
            exclude=np.number
        ).columns

    numeric_pipeline = Pipeline([

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

    categorical_pipeline = Pipeline([

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

    pipeline = ColumnTransformer([

        (

            "num",

            numeric_pipeline,

            numeric_cols

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_cols

        )

    ])

    return pipeline