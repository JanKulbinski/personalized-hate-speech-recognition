import pandas as pd

# METRICS
# columns: |topic	worker_id	0, 1, .... 13|

# metric 1
def avg_agression_per_topic(df_annotations_comments):
    return df_annotations_comments.pivot_table(
        index="worker_id",
        columns="topic",
        values="aggression",
        aggfunc="mean",
        fill_value=0.5,
    ).reset_index()  # .stack() for row-wise grouping


# metric 2
def avg_agression_score_per_topic(df_annotations_comments):
    return df_annotations_comments.pivot_table(
        index="worker_id",
        columns="topic",
        values="aggression_score",
        aggfunc="mean",
        fill_value=0,
    ).reset_index()


# metric 3
def avg_agression_score_per_topic_normalized(df_annotations_comments):
    df = df_annotations_comments.pivot_table(
        index="worker_id",
        columns="topic",
        values="aggression_score",
        aggfunc="mean",
        fill_value=0,
    ).reset_index()
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )
    return df


# metric 4
def mean_std_agression_score_per_topic(df_annotations_comments):
    df_mean = df_annotations_comments.pivot_table(
        index="worker_id",
        columns="topic",
        values="aggression_score",
        aggfunc="mean",
        fill_value=0,
    ).reset_index()

    df_std = df_annotations_comments.pivot_table(
        index="worker_id",
        columns="topic",
        values="aggression_score",
        aggfunc="std",
        fill_value=0,
    ).reset_index()

    df = pd.merge(df_mean, df_std, on="worker_id")

    return df


# metric 5 - ..... PEB,
