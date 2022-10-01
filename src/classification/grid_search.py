import torch

from load_data import (
    prepare_data_class_based,
    prepare_data_conformity_based,
    prepare_data_topic_based,
    reduce_set,
    to_data_loader,
)
from model import MLP_Model
from train import train_loop
from user_metrics import (
    avg_agression_per_topic,
    avg_agression_score_per_topic,
    avg_agression_score_per_topic_normalized,
    mean_std_agression_score_per_topic,
)
from tqdm import tqdm


user_metrics = [
    avg_agression_per_topic,
    avg_agression_score_per_topic,
    avg_agression_score_per_topic_normalized,
    mean_std_agression_score_per_topic,
]
lrs = [0.01, 0.001]
hidden_dims = [32, 64, 100]
batch_sizes = [64, 128]


def search_model(
    X_train, y_train, X_test, y_test, batch_size, optim_params, model_params
):
    X_test_reduced, y_test_reduced = reduce_set(X_test, y_test)

    train_dl = to_data_loader(X_train, y_train, batch_size)
    val_dl = to_data_loader(X_test_reduced, y_test_reduced, batch_size)
    test_dl = to_data_loader(X_test, y_test, batch_size)

    model = MLP_Model(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loop(
        model,
        train_dl,
        val_dl,
        test_dl,
        loss_fn,
        optimizer,
        epochs=100,
        patience=5,
    )


def grid_search(X_train, y_train, X_test, y_test, n_features, name, pbar=None):
    total_iters = len(lrs) * len(hidden_dims) * len(batch_sizes)

    def _search(pbar):
        for lr in lrs:
            for hidden in hidden_dims:
                for batch_size in batch_sizes:
                    optim_params = {"lr": lr}
                    model_params = {
                        "name": f"mlp-{name}-{lr}-{hidden}-{batch_size}",
                        "features": n_features,
                        "hidden": hidden,
                    }
                    search_model(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        batch_size,
                        optim_params,
                        model_params,
                    )
                    pbar.update(1)

    if pbar:
        _search(pbar)
    else:
        with tqdm(total=total_iters, desc="Grid search") as local_pbar:
            _search(local_pbar)

def grid_search_class_based():
    X_train, y_train, X_test, y_test = prepare_data_class_based()
    grid_search(X_train, y_train, X_test, y_test, 300, "class")

def grid_search_conformity_based():
    X_train, y_train, X_test, y_test = prepare_data_conformity_based()
    grid_search(X_train, y_train, X_test, y_test, 106, "conformity")


def grid_search_baseline():
    X_train, y_train, X_test, y_test = prepare_data_topic_based(is_baseline=True)
    grid_search(X_train, y_train, X_test, y_test, 100, "baseline")


def grid_search_topic_based():
    # algos = ["" for i in range(6)] + ["-hdbscan" for i in range(1)]
    # variants = ["macro" for i in range(3)]  + ["micro" for i in range(4)]
    # clusters = [2, 4, 6, 2, 5, 13, 3]
    
    algos = ["-hdbscan" for i in range(1)]
    variants = ["micro" for i in range(1)]
    clusters = [3]
    
    total_iters = (
        len(variants)
        * len(user_metrics)
        * len(lrs)
        * len(hidden_dims)
        * len(batch_sizes)
    )

    with tqdm(total=total_iters, desc="Grid search") as pbar:
        for variant, n_clusters, algo in zip(variants, clusters, algos):
            for func in user_metrics:
                X_train, y_train, X_test, y_test = prepare_data_topic_based(
                    n_clusters, func, variant, algo=algo, is_baseline=False
                )

                n_features = (
                    100 + int(n_clusters)
                    if func != mean_std_agression_score_per_topic
                    else 100 + 2 * int(n_clusters)
                )

                name = f"{variant}-{n_clusters}-{func.__name__}"
                grid_search(X_train, y_train, X_test, y_test, n_features, name, pbar)



if __name__ == '__main__':
    grid_search_topic_based()
    # grid_search_baseline()
    # grid_search_conformity_based()
    # grid_search_class_based()