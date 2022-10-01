import os
import torch
import pandas as pd

from sklearn.metrics import classification_report
from os.path import join
from consts import HOME_PATH
from load_data import prepare_data_topic_based
from model import MLP_Model

from user_metrics import (
    avg_agression_per_topic,
    avg_agression_score_per_topic,
    avg_agression_score_per_topic_normalized,
    mean_std_agression_score_per_topic,
)


best_hidden = 32
best_lr = 0.01
best_n_clusters = 2
best_func = avg_agression_per_topic
best_variant = "micro"


def print_report_for_model():
    checkpoint_path = join(
        os.getcwd(),
        f"{HOME_PATH}mlp/mlp-micro-2-avg_agression_per_topic-0.01-32-64.cpt",
    )

    model = MLP_Model(hidden=best_hidden, features=102)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    _, _, X_test, y_test = prepare_data_topic_based(
        best_n_clusters, avg_agression_per_topic, best_variant
    )

    y_pred = model(torch.Tensor(X_test)).cpu().detach().numpy().argmax(axis=1)

    print(classification_report(y_test, y_pred))


def add_column_model_type(df):
    df = df.reset_index()
    df = df.rename(columns={"index": "metric"})
    df["model_type"] = df["model"].str.split("-0", 1).str[0]
    return df


def get_best_models_names(df):
    # report_ones = df.loc[df['metric'] == '1', :]
    # report_ones.loc[report_ones.groupby(['model_type'])['recall'].idxmax()] # by recall-class-1
    df_accuracy = df.loc[df["metric"] == "accuracy", :]
    df_accuracy = df_accuracy.loc[df_accuracy.groupby(["model_type"])["precision"].idxmax()]
    best_models_list = df_accuracy["model"].tolist()
    return best_models_list


def get_best_results():  # best by f1-micro (= accuracy)
    results_all = pd.read_pickle(f"{HOME_PATH}mlp/results_df.pkl")
    results_all = add_column_model_type(results_all)

    best_models_names = get_best_models_names(results_all)
    bests_models = results_all.loc[results_all["model"].isin(best_models_names)]

    return bests_models


def results_for_latex_table(bests_models, metric, index_n):
    for variant in ["macro-2", "macro-4", "macro-6", "micro-2", "micro-5", "micro-13"]:
        res = ""
        for func in [
            avg_agression_per_topic,
            avg_agression_score_per_topic,
            mean_std_agression_score_per_topic,
        ]:
            res += f'\t{bests_models.loc[bests_models["model_type"] == f"mlp-{variant}-{func.__name__}", [metric]].iloc[index_n, 0]:.3f}'
        print(res)


# in latex-format
def print_bests_topic_based_results():
    bests_models = get_best_results()
    metrics = (
        ["precision" for _ in range(2)]
        + ["recall" for _ in range(2)]
        + ["f1-score" for _ in range(3)]
    )
    index_n = [3, 1, 3, 1, 2, 1, 3]
    names_sufix = ["macro", "class-1", "macro", "class-1", "micro", "class-1", "macro"]

    for metric, index, suffix in zip(metrics, index_n, names_sufix):
        print(f"{metric.upper()} {suffix}")
        results_for_latex_table(bests_models, metric, index)

def get_best_models_names_by_method_type(results_all, metric):
    results_all = add_column_model_type(results_all)

    results_all["method_type"] = results_all["model"].str.split("-",3).str[1]
    results_all.loc[results_all["method_type"].isin(["micro", "macro"]), "method_type"] = "topic-based"

    results_accuracy = results_all.loc[results_all["metric"] == metric, :]
    results_bests = results_accuracy.loc[results_accuracy.groupby(["method_type"])["f1-score"].idxmax()]
    best_models_names = results_bests["model"].tolist()  
    return best_models_names

# in latex-format
def print_bests_results(by_metric):
    results_all = pd.read_pickle(f"{HOME_PATH}mlp/results_df.pkl")
    bests_models_names = get_best_models_names_by_method_type(results_all, by_metric)
    
    metrics = (
        ["f1-score" for _ in range(3)]
        + ["precision" for _ in range(2)]
        + ["recall" for _ in range(2)]
    )
    index_n = [2, 3, 1, 3, 1, 3, 1]
    names_sufix = ["accuracy", "macro", "class-1", "macro", "class-1", "macro", "class-1"]
    
    [print(f'{metric.upper()} {suffix}') for metric, suffix in zip(metrics, names_sufix)]

    for model_name in bests_models_names:
        result = ''
        print(model_name)
        for metric, index in zip(metrics, index_n):
            result += f'\t{results_all.loc[results_all["model"] == model_name, [metric]].iloc[index, 0]:.3f}'

        print(result)

def best_models_parameters():
    results_all = pd.read_pickle(f"{HOME_PATH}mlp/results_df.pkl")
    results_all = add_column_model_type(results_all)

    best_models_names = get_best_models_names(results_all)[3:]
    del best_models_names[1::4]
    params = [model.split("-0")[1] for model in best_models_names]
    return params

if __name__ == "__main__":
    print_bests_topic_based_results()
    print(f'{"#" * 10}')
    print("BEST MODELS BY F1-micro")
    print_bests_results(by_metric="accuracy") # = accuracy


    print(f'{"#" * 10}')
    print("BEST MODELS BY F1-macro")
    print_bests_results(by_metric="macro avg") # = f1-macro 

    # there are best also regarding F1-macro  

