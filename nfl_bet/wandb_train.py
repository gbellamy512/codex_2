"""Training utilities with optional W&B integration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Dict
import pprint
import argparse

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .data_prep import prepare_df
from .betting import evaluate_betting_strategy, get_betting_context

try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
except Exception:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore
    WandbMetricsLogger = None  # type: ignore


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
FIRST_YEAR = 2013
CURRENT_YEAR = 2024


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def load_data():
    data_hist = load_csv(f"ewma_epa_df_2006_{CURRENT_YEAR-1}.csv")
    data_cy = load_csv(f"ewma_epa_df_{CURRENT_YEAR}.csv")
    data = pd.concat([data_hist, data_cy])

    pass_rates_hist = load_csv(f"pass_rates_{FIRST_YEAR}_{CURRENT_YEAR-1}.csv")
    pass_rates_ancient = load_csv("pass_rates_2006_2012.csv")
    pass_rates_cy = load_csv(f"pass_rates_{CURRENT_YEAR}.csv")
    pass_rates = pd.concat([pass_rates_ancient, pass_rates_hist, pass_rates_cy])

    win_hist = load_csv(f"weekly_records_1999_{CURRENT_YEAR-1}.csv")
    win_cy = load_csv(f"weekly_records_{CURRENT_YEAR}.csv")
    win_percentages = pd.concat([win_hist, win_cy])

    sched_hist = load_csv(f"nfl_schedules_1999_{CURRENT_YEAR-1}.csv")
    sched_cy = load_csv(f"nfl_schedules_{CURRENT_YEAR}.csv")
    schedules = pd.concat([sched_hist, sched_cy])

    return data, pass_rates, win_percentages, schedules


# ---------------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------------

features = [
    "rushing_offense_adv",
    "passing_offense_adv",
    "rushing_defense_adv",
    "passing_defense_adv",
    "win_percentage_diff",
    "implied_prob_diff",
    "rest_advantage",
    "div_game",
    "h2h_type",
]

categorical_features = ["h2h_type"]
binary_features = ["div_game"]
non_numerical_features = categorical_features + binary_features
numerical_features = [f for f in features if f not in non_numerical_features]

def get_activation(name: str):
    if name == "elu":
        return tf.keras.activations.elu
    return name


def evaluate_and_log_metrics(
    model, X, y, tag: str, model_type: str
) -> Dict[str, float]:
    """Evaluate a model on ``X``/``y`` and log metrics via W&B.

    Parameters
    ----------
    model : keras.Model
        The model to evaluate.
    X, y : array-like
        Evaluation data.
    tag : str
        Prefix used when logging metrics.
    model_type : {"classification", "regression"}
        Determines which metrics to expect and log.
    """

    results = model.evaluate(X, y, verbose=0)

    if model_type == "classification":
        loss, accuracy, precision, recall = results
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall
            else 0
        )
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "len": len(X),
        }
    else:  # regression
        loss, mse = results
        metrics = {"loss": loss, "mse": mse, "len": len(X)}

    if wandb is not None:
        wandb.log({f"{tag}_" + k: v for k, v in metrics.items()})

    return metrics


def evaluate_and_log_metrics_betting(
    dataset: pd.DataFrame,
    prefix: str,
    model,
    pipeline,
    *,
    bet_strat: str,
    margin: float,
    orientation: str,
    bet_type: str,
) -> None:
    ctx = get_betting_context(orientation, bet_type)
    model_type = "regression" if bet_type == "spread" else "classification"
    results = evaluate_betting_strategy(
        dataset,
        model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin,
        target=ctx["target"],
        team1_label=ctx["team1_label"],
        team2_label=ctx["team2_label"],
        team1_odds_col=ctx["team1_odds_col"],
        team2_odds_col=ctx["team2_odds_col"],
        model_type=model_type,
        line_col=ctx.get("line_col"),
    )
    if wandb is not None:
        wandb.log({
            f"{prefix}_profit": results["total_profit"],
            f"{prefix}_bet": results["total_bet"],
            f"{prefix}_roi": results["roi"],
        })


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: Optional[dict] = None) -> None:
    if wandb is None:
        raise ImportError("wandb must be installed to use the train function")

    project = None
    if config is not None:
        project = config.get("project")

    with wandb.init(project=project, config=config):
        cfg = wandb.config
        orientation = getattr(cfg, "orientation", "fav_dog")
        bet_type = getattr(cfg, "bet_type", "moneyline")
        model_type = "regression" if bet_type == "spread" else "classification"
        wandb.config.update({"model_type": model_type}, allow_val_change=True)
        wandb.log({"orientation": orientation, "bet_type": bet_type, "model_type": model_type})

        data, pass_rates, win_percentages, schedules = load_data()
        df = prepare_df(
            data=data,
            pass_rates=pass_rates,
            win_percentages=win_percentages,
            schedules=schedules,
            min_periods=cfg.min_periods,
            span=cfg.span,
            avg_method="simple",
            orientation=orientation,
            bet_type=bet_type,
        )

        ctx = get_betting_context(orientation, bet_type)

        cy_df = df[df["season"] == CURRENT_YEAR]
        df = df[df["season"] < CURRENT_YEAR]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features + binary_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )

        # Fresh metric instances for each run to avoid duplicate logging
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()

        random_state = random.randint(0, 1000)
        wandb.log({"random_state": random_state})

        target_col = ctx["regression_target"] if model_type == "regression" else ctx["target"]
        stratify_col = df[target_col] if model_type == "classification" else None
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            df[features],
            df[target_col],
            test_size=cfg.test_size,
            stratify=stratify_col,
            random_state=random_state,
        )
        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            X_train_df,
            y_train_df,
            test_size=cfg.test_size,
            stratify=y_train_df if model_type == "classification" else None,
            random_state=random_state,
        )

        pipeline = Pipeline([("preprocessor", preprocessor)])
        X_train = pipeline.fit_transform(X_train_df)
        X_val = pipeline.transform(X_val_df)
        X_test = pipeline.transform(X_test_df)

        pipeline_file = "pipeline.pkl"
        joblib.dump(pipeline, pipeline_file)
        artifact = wandb.Artifact(f"preprocessing_pipeline_{wandb.run.id}", type="pipeline")
        artifact.add_file(pipeline_file)
        wandb.log_artifact(artifact)

        y_train = y_train_df.values
        y_val = y_val_df.values
        y_test = y_test_df.values

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
        for _ in range(cfg.hidden_layers):
            if cfg.batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(cfg.dropout))
            model.add(
                tf.keras.layers.Dense(
                    cfg.neurons,
                    activation=get_activation(cfg.activation),
                    kernel_initializer=cfg.kernel_initializer,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=cfg.l1_reg, l2=cfg.l2_reg),
                )
            )
        model.add(tf.keras.layers.Dropout(cfg.dropout))
        final_activation = None if model_type == "regression" else cfg.outer_activation
        model.add(tf.keras.layers.Dense(1, activation=final_activation))

        if cfg.optimizer == "sgd":
            momentum_value = float(cfg.sgd_momentum)
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=cfg.learning_rate, momentum=momentum_value
            )
        elif cfg.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        elif cfg.optimizer == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        if model_type == "regression":
            loss = "mse"
            metrics = ["mse"]
        else:
            loss = cfg.loss
            metrics = [cfg.metric, precision_metric, recall_metric]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        callbacks = [WandbMetricsLogger()]

        if cfg.early_stopping:
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                restore_best_weights=True,
            )
            callbacks.append(early_stopping_cb)

        if cfg.apply_lr_schedule:
            def scheduler(epoch, lr):
                if (epoch + 1) % cfg.lr_scheduler_step_every == 0:
                    return lr * cfg.lr_scheduler_step_factor
                return lr

            lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callbacks.append(lr_scheduler_cb)

        if model_type == "classification" and cfg.apply_class_weights:
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None

        model.fit(
            x=X_train,
            y=y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight_dict,
        )

        model_file = "model.keras"
        model.save(model_file)
        artifact = wandb.Artifact(f"model_{wandb.run.id}", type="model")
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)

        evaluate_and_log_metrics(model, X_train, y_train, "train", model_type)
        evaluate_and_log_metrics(model, X_val, y_val, "val", model_type)
        evaluate_and_log_metrics(model, X_test, y_test, "test", model_type)

        X_cy = cy_df[features]
        y_cy = cy_df[target_col]
        X_cy = pipeline.transform(X_cy)
        y_cy = y_cy.values
        evaluate_and_log_metrics(model, X_cy, y_cy, "cy", model_type)


# ---------------------------------------------------------------------------
# Sweep utilities
# ---------------------------------------------------------------------------

def create_sweep(
    project: str,
    *,
    orientation: str = "fav_dog",
    bet_type: str = "moneyline",
) -> str:
    """Create a W&B sweep using hyperparameter ranges.

    The configuration mirrors the sweep setup from the original notebook.
    ``project`` must be provided explicitly.
    """

    if wandb is None:
        raise ImportError("wandb must be installed to create a sweep")

    project_name = project

    sweep_config: Dict[str, object] = {
        "method": "bayes",
    }
    sweep_config["metric"] = {"name": "val_loss", "goal": "minimize"}

    parameters_dict = {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "batch_size": {"values": [16, 32, 64]},
        "hidden_layers": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 4,
        },
        "neurons": {"values": [16, 32, 64]},
        "batch_norm": {"values": [True, False]},
        "l1_reg": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "l2_reg": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5,
        },
        "apply_class_weights": {"values": [True, False]},
        "early_stopping_patience": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 35,
        },
        "early_stopping_min_delta": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "lr_scheduler_step_every": {
            "distribution": "int_uniform",
            "min": 5,
            "max": 30,
        },
        "lr_scheduler_step_factor": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.8,
        },
    }

    sweep_config["parameters"] = parameters_dict

    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_name = gpu_devices[0].name if gpu_devices else "CPU"

    parameters_dict.update(
        {
            "gpu_name": {"value": gpu_name},
            "orientation": {"value": orientation},
            "bet_type": {"value": bet_type},
            "model_type": {"value": "regression" if bet_type == "spread" else "classification"},
            "loss": {"value": "mse" if bet_type == "spread" else "binary_crossentropy"},
            "metric": {"value": "mse" if bet_type == "spread" else "accuracy"},
            "optimizer": {"value": "adam"},
            "kernel_initializer": {"value": "he_normal"},
            "activation": {"value": "relu"},
            "outer_activation": {"value": "linear" if bet_type == "spread" else "sigmoid"},
            "epochs": {"value": 200},
            "early_stopping": {"value": True},
            "apply_lr_schedule": {"value": True},
            "test_size": {"value": 0.1},
            "min_periods": {"value": 3},
            "span": {"value": 8},
        }
    )

    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    return sweep_id


def run_sweep(sweep_id: str, *, count: int = 50) -> None:
    """Launch a W&B sweep agent."""

    if wandb is None:
        raise ImportError("wandb must be installed to run sweeps")

    wandb.agent(sweep_id=sweep_id, function=train, count=count)

# ---------------------------------------------------------------------------
# Example entry point
# ---------------------------------------------------------------------------

def example_run(project: str, orientation: str = "fav_dog", bet_type: str = "moneyline") -> None:
    if wandb is None:
        raise ImportError("wandb must be installed to run the example")

    default_config = {
        "project": project,
        "orientation": orientation,
        "bet_type": bet_type,
        "model_type": "regression" if bet_type == "spread" else "classification",
        "test_size": 0.2,
        "epochs": 1,
        "batch_size": 64,
        "hidden_layers": 2,
        "neurons": 32,
        "dropout": 0.2,
        "activation": "relu",
        "outer_activation": "sigmoid",
        "optimizer": "adam",
        "sgd_momentum": 0.0,
        "learning_rate": 0.001,
        "loss": "binary_crossentropy",
        "metric": "accuracy",
        "batch_norm": False,
        "kernel_initializer": "glorot_uniform",
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "early_stopping": False,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "apply_lr_schedule": False,
        "lr_scheduler_step_every": 10,
        "lr_scheduler_step_factor": 0.1,
        "apply_class_weights": False,
        "min_periods": 3,
        "span": 8,
        "bet_strat": "both",
        "margin": 0.0,
    }
    if bet_type == "spread":
        default_config["outer_activation"] = "linear"
        default_config["loss"] = "mse"
        default_config["metric"] = "mse"
        default_config["model_type"] = "regression"
    train(default_config)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI usage."""

    parser = argparse.ArgumentParser(description="NFL betting training utils")
    sub = parser.add_subparsers(dest="command", required=True)

    ex_p = sub.add_parser("example", help="run the example training run")
    ex_p.add_argument(
        "--project",
        required=True,
        help="W&B project name",
    )
    ex_p.add_argument(
        "--orientation",
        choices=["fav_dog", "home_away"],
        default="fav_dog",
    )
    ex_p.add_argument(
        "--bet-type",
        choices=["moneyline", "spread"],
        default="moneyline",
        dest="bet_type",
    )

    sweep_parser = sub.add_parser(
        "sweep", help="create a hyperparameter sweep and run it"
    )
    sweep_parser.add_argument(
        "--project",
        required=True,
        help="W&B project name",
    )
    sweep_parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of sweep runs to execute",
    )
    sweep_parser.add_argument(
        "--orientation",
        choices=["fav_dog", "home_away"],
        default="fav_dog",
    )
    sweep_parser.add_argument(
        "--bet-type",
        choices=["moneyline", "spread"],
        default="moneyline",
        dest="bet_type",
    )

    args = parser.parse_args(argv)

    if args.command == "example":
        example_run(
            project=args.project,
            orientation=args.orientation,
            bet_type=args.bet_type,
        )
    elif args.command == "sweep":
        sid = create_sweep(
            project=args.project, orientation=args.orientation, bet_type=args.bet_type
        )
        run_sweep(sid, count=args.count)


if __name__ == "__main__":  # pragma: no cover
    main()
