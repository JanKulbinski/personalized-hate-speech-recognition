# import pickle5 as pickle
import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from os.path import exists

from train import HOME_PATH
from user_metrics import avg_agression_per_topic


def read_embeddings():
    if exists(f"{HOME_PATH}df_embeddings_splitted.pkl"):
        df_embedding_splited = pd.read_pickle(f"{HOME_PATH}df_embeddings_splitted.pkl")
    else:
        with open(HOME_PATH + "df_comments_embeddings.pkl", "rb") as f:
            df_embedding = pickle.load(f)
            df_embedding.rename(columns={"comment": "embedding"}, inplace=True)
            df_embedding = df_embedding[["rev_id", "embedding"]]
        df_embedding_splited = df_embedding.join(
            pd.DataFrame(
                df_embedding.pop("embedding").to_list(), columns=list(range(100))
            )
        )
        df_embedding_splited.to_pickle(f"{HOME_PATH}df_embeddings_splitted.pkl")
    return df_embedding_splited


def read_annotation_comment(n_clusters, name, variant, algo):
    df_annotations = pd.read_pickle(f"{HOME_PATH}df_anno_{name}.pkl")
    df_comments = pd.read_pickle(
        f"{HOME_PATH}df-comments-clusters{algo}-{variant}-{n_clusters}-{name}.pkl"
    )
    print(df_comments)
    df_anno_comm = pd.merge(df_annotations, df_comments, on="rev_id")

    return df_anno_comm


# def get_X_y_2 .... join only metric which represents current topic and not all
def join_metric_to_annotation(n_clusters, metric_func, variant, algo):

    # worker metrics
    df_annotations_comments_dev = read_annotation_comment(n_clusters, "dev", variant, algo)
    metrics_df = metric_func(df_annotations_comments_dev)
    new_columns = {i: f"topic_{i}" for i in range(n_clusters)}
    metrics_df.rename(columns=new_columns, inplace=True)

    # join with annotation
    def join(metrics_df, name, n_clusters):
        df_annotations = read_annotation_comment(n_clusters, name, variant, algo)
        df_anno_metric = pd.merge(df_annotations, metrics_df, on="worker_id")
        return df_anno_metric

    df_annotations_train = join(metrics_df, "train", n_clusters)
    df_annotations_test = join(metrics_df, "test", n_clusters)
    return df_annotations_train, df_annotations_test


def join_embedding(df):
    df_embedding = read_embeddings()
    return pd.merge(df, df_embedding, on="rev_id")


def prepare_data_topic_based(
    n_clusters=2,
    metric_func=avg_agression_per_topic,
    variant="micro",
    is_baseline=False,
    algo=""
):
    df_anno_metric_train, df_anno_metric_test = join_metric_to_annotation(
        n_clusters, metric_func, variant, algo
    )
    print(n_clusters)

    df_train = join_embedding(df_anno_metric_train)
    df_test = join_embedding(df_anno_metric_test)

    def get_X_y(df):
        if is_baseline:
            features_start = 10 + n_clusters
        else:
            features_start = 10
        X = df.iloc[:, features_start:].to_numpy()
        y = df.iloc[:, 2].to_numpy()
        return X, y

    X_train, y_train = get_X_y(df_train)
    X_test, y_test = get_X_y(df_test)


    return X_train, y_train, X_test, y_test


def load_data_conformity_based(name: str):
    data_path = os.path.join(HOME_PATH, "conformity-based", f"{name}_data.tsv")
    labels_path = os.path.join(HOME_PATH, "conformity-based", f"{name}_labels.tsv")
    df_X = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=[
            "rev_id",
            "gconf_0",
            "gconf_1",
            "gconf_01",
            "wconf_0",
            "wconf_1",
            "wconf_01",
        ],
    )

    df_X = join_embedding(df_X)
    X = df_X.iloc[:, 1:].to_numpy()

    y = []
    with open(labels_path, "r") as f:
        for line in f.readlines():
            y.append(int(float(line.strip())))

    return X, np.array(y)


def prepare_data_conformity_based():
    X_train, y_train = load_data_conformity_based("train")
    X_test, y_test = load_data_conformity_based("test")

    return X_train, y_train, X_test, y_test


def class_based_workers_metrics():
    df_annotations = pd.read_pickle(f"{HOME_PATH}df_anno_dev.pkl")
    df_annotations = join_embedding(df_annotations)

    df_annotations_slim = df_annotations.drop(df_annotations.columns[[0, 3]], axis=1)
    df_workers_mean_anno = (
        df_annotations_slim.pivot_table(
            index="worker_id", columns="aggression", aggfunc="mean", fill_value=0
        )
        .stack()
        .reset_index()
    )
    df_workers_mean_anno_0 = df_workers_mean_anno.loc[
        df_workers_mean_anno["aggression"] == 0
    ]
    df_workers_mean_anno_0 = df_workers_mean_anno_0.drop("aggression", axis=1)

    df_workers_mean_anno_1 = df_workers_mean_anno.loc[
        df_workers_mean_anno["aggression"] == 1
    ]
    df_workers_mean_anno_1 = df_workers_mean_anno_1.drop("aggression", axis=1)

    return df_workers_mean_anno_0, df_workers_mean_anno_1


def prepare_data_class_based():
    df_workers_mean_anno_0, df_workers_mean_anno_1 = class_based_workers_metrics()

    def _join_annotations_to_metrics(name):
        df_annotations = pd.read_pickle(f"{HOME_PATH}df_anno_{name}.pkl")
        df_annotations = join_embedding(df_annotations)
        df_annotations_workers = df_annotations.merge(
            df_workers_mean_anno_0, on="worker_id"
        )
        df_annotations_workers = df_annotations_workers.merge(
            df_workers_mean_anno_1, on="worker_id"
        )

        labels = df_annotations_workers["aggression"].to_numpy()
        df_annotations_workers = df_annotations_workers.iloc[:, 4:].to_numpy()

        return df_annotations_workers, labels

    X_train, y_train = _join_annotations_to_metrics("train")
    X_test, y_test = _join_annotations_to_metrics("test")

    return X_train, y_train, X_test, y_test


def reduce_set(X, y, size=60000):
    _, X_reduced, _, y_reduced = train_test_split(
        X, y, test_size=size, random_state=42, stratify=y
    )

    return X_reduced, y_reduced


def to_data_loader(X, y, batch_size):
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y).long()

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
