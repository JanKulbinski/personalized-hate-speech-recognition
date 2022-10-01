import torch
import functools
import operator
import os
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from typing import Any, Tuple
from tqdm.auto import tqdm
from os.path import join
from os.path import exists
from consts import DEVICE, HOME_PATH


from model import MLP_Model


def train_loop(
    model: MLP_Model,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    epochs: int = 200,
    patience: int = 10,
):

    checkpoint_path = join(os.getcwd(), f"{HOME_PATH}/mlp/{model.name}.cpt")
    _train_loop(
        model.to(DEVICE),
        train_dl,
        val_dl,
        test_dl,
        loss_fn,
        optimizer,
        checkpoint_path,
        epochs=epochs,
        patience=patience,
    )


def _train_loop(
    model: MLP_Model,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    *,
    epochs: int = 200,
    patience: int = 10,
):
    best_val_set_loss = float("inf")
    no_impr_iters = 0
    for epoch in tqdm(range(epochs), desc="Epoch", leave=False):

        train_step(model, train_dl, loss_fn, optimizer)
        val_loss = evaluation_step(model, val_dl, loss_fn)

        if best_val_set_loss > val_loss:
            best_val_set_loss = val_loss
            torch.save(
                obj={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f=checkpoint_path,
            )
            no_impr_iters = 0
        else:
            no_impr_iters += 1

        if no_impr_iters > patience:
            break

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    save_models_report(model, test_dl)


def save_models_report(
    model: MLP_Model,
    test_dl: DataLoader,
):
    y_all_true = []
    y_all_pred = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)

            y_all_true.append(y_batch.cpu().detach().numpy().tolist())
            y_all_pred.append(y_pred.cpu().detach().numpy().argmax(axis=1).tolist())

    y_all_true = functools.reduce(operator.iconcat, y_all_true, [])
    y_all_pred = functools.reduce(operator.iconcat, y_all_pred, [])

    if exists(f"{HOME_PATH}mlp/results_df.pkl"):
        report_all = pd.read_pickle(f"{HOME_PATH}mlp/results_df.pkl")
    else:
        report_all = pd.DataFrame([])

    report = classification_report(y_all_true, y_all_pred, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df["model"] = model.name
    report_all = pd.concat([report_all, df])
    report_all.to_pickle(f"{HOME_PATH}mlp/results_df.pkl")


def train_step(
    model: MLP_Model,
    train_dl: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    for X, y in train_dl:
        model.zero_grad()

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def evaluator(
    model: MLP_Model, data_loader: DataLoader, loss_fn: Any, tag: Any
) -> None:

    preds_counter = 0
    loss_all = 0
    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch

        y_pred = model(X_batch).to("cpu")
        loss_all += loss_fn(y_pred, y_batch).sum()
        preds_counter += len(y_pred)

    loss_all = (loss_all / preds_counter).cpu().item()

    return loss_all


def evaluation_step(
    model: MLP_Model,
    val_dl: DataLoader,
    loss_fn: Any,
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        loss_test = evaluator(model, val_dl, loss_fn, tag="test")

    return loss_test
