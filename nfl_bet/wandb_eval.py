"""Utilities for evaluating W&B runs using betting metrics."""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from joblib import load

from . import evaluate_betting_strategy, prepare_df, get_betting_context
from .wandb_train import load_data, CURRENT_YEAR

RESULTS_DIR = "results"

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional wandb
    wandb = None  # type: ignore


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_top_runs_quickly(
    wandb_project: str,
    metric_threshold: float,
    metric: str = "loss",
    top_n: Optional[int] = None,
    train_weight: float = 1.0,
) -> pd.DataFrame:
    """Return metadata for the top runs of a W&B project."""
    if wandb is None:
        raise RuntimeError("wandb is required for get_top_runs_quickly")

    allowed = {"loss", "accuracy", "precision", "recall", "f1"}
    if metric not in allowed:
        raise ValueError(f"Invalid metric '{metric}'")
    lowest_is_better = metric == "loss"

    api = wandb.Api()
    runs = api.runs(wandb_project)

    summaries = []
    for run in runs:
        s = run.summary._json_dict
        summaries.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                f"train_{metric}": s.get(f"train_{metric}"),
                f"val_{metric}": s.get(f"val_{metric}"),
                f"test_{metric}": s.get(f"test_{metric}"),
            }
        )
    df = pd.DataFrame(summaries).dropna(
        subset=[f"train_{metric}", f"val_{metric}", f"test_{metric}"]
    )

    df["aggregate_metric"] = (
        train_weight * df[f"train_{metric}"]
        + df[f"val_{metric}"]
        + df[f"test_{metric}"]
    )
    df = df.sort_values("aggregate_metric", ascending=lowest_is_better)

    total_weight = train_weight + 2
    cutoff = metric_threshold * total_weight
    if lowest_is_better:
        df = df[df["aggregate_metric"] <= cutoff]
    else:
        df = df[df["aggregate_metric"] >= cutoff]
    print(f"Filtered to {len(df)} runs.")

    if top_n:
        if lowest_is_better:
            df = df.nsmallest(top_n, "aggregate_metric")
        else:
            df = df.nlargest(top_n, "aggregate_metric")

    details = []
    for _, row in df.iterrows():
        run = api.run(f"{wandb_project}/{row['run_id']}")
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        arts = [a.name for a in run.logged_artifacts() if "-history" not in a.name]
        details.append(
            {
                **summary,
                **config,
                "run_id": run.id,
                "run_name": run.name,
                "last_artifact_name": arts[-1] if arts else None,
            }
        )
    return pd.DataFrame(details)


def load_model_and_pipeline(
    run_id: str,
    last_artifact_name: str,
    wandb_project: str,
    entity: Optional[str] = os.getenv("WANDB_ENTITY"),
) -> Tuple[tf.keras.Model, object]:
    """Load a model and preprocessing pipeline from W&B."""
    if wandb is None:
        raise RuntimeError("wandb is required for load_model_and_pipeline")

    entity = entity or os.getenv("WANDB_ENTITY")
    prefix = f"{entity}/" if entity else ""

    api = wandb.Api()
    model_art = api.artifact(f"{prefix}{wandb_project}/{last_artifact_name}")
    model_dir = model_art.download()
    model = tf.keras.models.load_model(f"{model_dir}/model.keras")

    pipe_art = api.artifact(
        f"{prefix}{wandb_project}/preprocessing_pipeline_{run_id}:v0"
    )
    pipe_dir = pipe_art.download()
    pipeline = load(f"{pipe_dir}/pipeline.pkl")
    print(last_artifact_name)
    return model, pipeline


def evaluate_betting_results(
    model,
    pipeline,
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    bet_strat: str,
    margin: float,
    orientation: str,
    bet_type: str,
    cy_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Return betting metrics for train/val/test and optional current year."""
    ctx = get_betting_context(orientation, bet_type)
    model_type = "regression" if bet_type == "spread" else "classification"
    train_res = evaluate_betting_strategy(
        df=df_train,
        model=model,
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
    train_metrics = {
        "train_profit": train_res["total_profit"],
        "train_bet": train_res["total_bet"],
        "train_roi": train_res["roi"],
        "train_bet_rate": train_res["bet_rate"],
    }

    val_res = evaluate_betting_strategy(
        df=df_val,
        model=model,
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
    val_metrics = {
        "val_profit": val_res["total_profit"],
        "val_bet": val_res["total_bet"],
        "val_roi": val_res["roi"],
        "val_bet_rate": val_res["bet_rate"],
    }

    test_res = evaluate_betting_strategy(
        df=df_test,
        model=model,
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
    test_metrics = {
        "test_profit": test_res["total_profit"],
        "test_bet": test_res["total_bet"],
        "test_roi": test_res["roi"],
        "test_bet_rate": test_res["bet_rate"],
    }

    cy_metrics = {}
    if cy_df is not None:
        cy_res = evaluate_betting_strategy(
            df=cy_df,
            model=model,
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
        cy_metrics = {
            "cy_profit": cy_res["total_profit"],
            "cy_bet": cy_res["total_bet"],
            "cy_roi": cy_res["roi"],
            "cy_bet_rate": cy_res["bet_rate"],
        }

    return {**train_metrics, **val_metrics, **test_metrics, **cy_metrics}


def exe(
    df_runs_epochs: pd.DataFrame,
    features: List[str],
    wandb_project: str,
    bet_strats: Optional[List[str]] = None,
    margins: Optional[List[float]] = None,
    cy_df: Optional[pd.DataFrame] = None,
    exclude_tested: bool = True,
    pull_high_roi: bool = False,
    entity: Optional[str] = os.getenv("WANDB_ENTITY"),
    orientation: str = "fav_dog",
    bet_type: str = "moneyline",
) -> pd.DataFrame:
    """Evaluate each run for all strategy/margin combinations."""
    if wandb is None:
        raise RuntimeError("wandb is required for exe")

    if bet_strats is None:
        bet_strats = (
            ["both", "home", "away"]
            if orientation == "home_away"
            else ["both", "dog", "fav"]
        )
    if margins is None:
        margins = [0, 0.5, 1, 1.5, 2] if bet_type == "spread" else [0.025, 0.05, 0.075, 0.1]

    data, pass_rates, win_percentages, schedules = load_data()

    results_list = []
    api = wandb.Api()

    for _, row in df_runs_epochs.iterrows():
        run_name = row["run_name"]
        run_id = row["run_id"]
        last_artifact_name = row["last_artifact_name"]
        print(f"\nProcessing run '{run_name}' (ID: {run_id})...")

        run_obj = None
        if exclude_tested or pull_high_roi:
            try:
                run_obj = api.run(f"{wandb_project}/{run_id}")
            except wandb.errors.CommError as e:
                print(
                    f"Error fetching run {run_id} from project {wandb_project}. Skipping. Error: {e}"
                )
                continue

        if exclude_tested and run_obj and "tested" in run_obj.tags:
            print(f"Skipping already tested run '{run_name}' (ID: {run_id}).")
            continue

        if "history" in str(last_artifact_name).lower():
            print(
                f"Skipping run '{run_name}' with artifact '{last_artifact_name}' (contains 'history')."
            )
            continue

        high_roi_combos_to_evaluate: set[Tuple[str, float]] = set()
        if pull_high_roi:
            if not run_obj:
                print(
                    f"Error: Could not fetch run object for {run_id} when pull_high_roi=True. Skipping."
                )
                continue
            tag_pattern = re.compile(r"high_roi_S_(\w+)_M_(\d+)")
            for tag in run_obj.tags:
                match = tag_pattern.match(tag)
                if match:
                    strategy = match.group(1)
                    margin_value = int(match.group(2))
                    margin_float = margin_value / 1000.0
                    high_roi_combos_to_evaluate.add((strategy, margin_float))
                    print(
                        f"  Found high ROI combo from tag '{tag}': Strategy='{strategy}', Margin={margin_float}"
                    )
            if not high_roi_combos_to_evaluate:
                print(
                    f"Skipping run '{run_name}' (ID: {run_id}) as pull_high_roi=True and no tags found."
                )
                continue

        try:
            model, pipeline = load_model_and_pipeline(
                run_id,
                last_artifact_name,
                wandb_project,
                entity=entity,
            )
        except Exception as e:
            print(f"Error loading model/pipeline for run {run_id}. Skipping. Error: {e}")
            continue

        random_state = row["random_state"]
        min_periods = row["min_periods"]
        span = row["span"]
        print(
            f"  Run Params: random_state={random_state}, min_periods={min_periods}, span={span}"
        )

        orientation_run = row.get("orientation", orientation)
        bet_type_run = row.get("bet_type", bet_type)
        ctx = get_betting_context(orientation_run, bet_type_run)
        df_prepared = prepare_df(
            data=data,
            pass_rates=pass_rates,
            win_percentages=win_percentages,
            schedules=schedules,
            min_periods=min_periods,
            span=span,
            orientation=orientation_run,
            bet_type=bet_type_run,
        )
        cy = CURRENT_YEAR
        current_cy_df = df_prepared[df_prepared["season"] == cy].copy()
        df_prepared = df_prepared[df_prepared["season"] < cy]

        target = ctx["target"]
        test_size = 0.1
        if df_prepared.empty or len(df_prepared[target].unique()) < 2:
            print(
                f"Skipping run {run_id} due to insufficient data or classes for splitting after filtering."
            )
            continue

        X = df_prepared[features]
        y = df_prepared[target]
        if y.value_counts().min() < 2:
            print(
                f"Skipping run {run_id}: Not enough samples in the smallest class for stratification."
            )
            continue

        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        if y_train_df.value_counts().min() < 2:
            print(
                f"Skipping run {run_id}: Not enough samples in the smallest class for validation split stratification."
            )
            continue
        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            X_train_df,
            y_train_df,
            test_size=test_size,
            stratify=y_train_df,
            random_state=random_state,
        )

        df_train = df_prepared.loc[X_train_df.index].copy()
        df_val = df_prepared.loc[X_val_df.index].copy()
        df_test = df_prepared.loc[X_test_df.index].copy()

        combos = (
            high_roi_combos_to_evaluate if pull_high_roi else [(s, m) for s in bet_strats for m in margins]
        )
        if pull_high_roi:
            print(f"  Evaluating {len(combos)} high ROI combination(s) found in tags...")
        else:
            print(f"  Evaluating all {len(bet_strats)} strategies and {len(margins)} margins...")

        run_evaluated = False
        for bet_strat, margin in combos:
            print(f"    Evaluating: Strategy='{bet_strat}', Margin={margin}")
            try:
                eval_dict = evaluate_betting_results(
                    model=model,
                    pipeline=pipeline,
                    features=features,
                    df_train=df_train,
                    df_val=df_val,
                    df_test=df_test,
                    bet_strat=bet_strat,
                    margin=margin,
                    orientation=orientation_run,
                    bet_type=bet_type_run,
                    cy_df=current_cy_df,
                )
                eval_dict.update(
                    {
                        "bet_strat": bet_strat,
                        "margin": margin,
                        "run_id": run_id,
                        "last_artifact_name": last_artifact_name,
                        "run_name": run_name,
                    }
                )
                results_list.append(eval_dict)
                run_evaluated = True
            except Exception as e:
                print(f"    Error during evaluation for combo (Strat: {bet_strat}, Margin: {margin}): {e}")

        if run_evaluated and run_obj and "tested" not in run_obj.tags:
            try:
                print(f"  Marking run '{run_name}' as 'tested'.")
                run_obj.tags.append("tested")
                run_obj.update()
            except Exception as e:
                print(f"  Error adding 'tested' tag to run '{run_name}': {e}")

    if not results_list:
        print("\nNo results were generated.")
        return pd.DataFrame()
    print(f"\nFinished processing. Combining {len(results_list)} results.")
    return pd.DataFrame(results_list)


def run_pipeline(
    wandb_project: str,
    features: List[str],
    top_metric: str = "loss",
    top_n: int = 10,
    train_weight: float = 1.0,
    bet_strats: Optional[List[str]] = None,
    margins: Optional[List[float]] = None,
    cy_df: Optional[pd.DataFrame] = None,
    exclude_tested: bool = True,
    pull_high_roi: bool = False,
    metric_threshold: Optional[float] = None,
    entity: Optional[str] = os.getenv("WANDB_ENTITY"),
    orientation: str = "fav_dog",
    bet_type: str = "moneyline",
) -> pd.DataFrame:
    """Fetch top runs and evaluate betting ROI."""
    if bet_type == "spread":
        if margins is None:
            margins = [0, 0.5, 1, 1.5, 2]
        metric_threshold = 175.0 if metric_threshold is None else metric_threshold
    else:
        if margins is None:
            margins = [0.025, 0.05, 0.075, 0.1]
        metric_threshold = 0.60 if metric_threshold is None else metric_threshold

    top_runs_df = get_top_runs_quickly(
        wandb_project=wandb_project,
        metric=top_metric,
        top_n=top_n,
        train_weight=train_weight,
        metric_threshold=metric_threshold,
    )
    if bet_strats is None:
        bet_strats = (
            ["both", "home", "away"]
            if orientation == "home_away"
            else ["both", "dog", "fav"]
        )

    results_df = exe(
        df_runs_epochs=top_runs_df,
        features=features,
        wandb_project=wandb_project,
        bet_strats=bet_strats,
        margins=margins,
        cy_df=cy_df,
        exclude_tested=exclude_tested,
        pull_high_roi=pull_high_roi,
        entity=entity,
        orientation=orientation,
        bet_type=bet_type,
    )
    return results_df


def evaluate_single_run(
    run_id: str,
    features: List[str],
    bet_strat: str,
    margin: float,
    wandb_project: str,
    entity: Optional[str] = os.getenv("WANDB_ENTITY"),
    cy_year: int = CURRENT_YEAR,
    orientation: str = "fav_dog",
    bet_type: str = "moneyline",
) -> Tuple[dict, dict]:
    """Evaluate a single run for one strategy and margin."""
    if wandb is None:
        raise RuntimeError("wandb is required for evaluate_single_run")

    api = wandb.Api()
    entity = entity or os.getenv("WANDB_ENTITY")
    prefix = f"{entity}/" if entity else ""
    run = api.run(f"{prefix}{wandb_project}/{run_id}")
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    details = {**summary, **config}
    required = ["random_state", "min_periods", "span"]
    missing = [k for k in required if k not in details]
    if missing:
        raise KeyError(
            f"Run '{run_id}' missing required key(s): {missing}. Available in summary+config: {list(details.keys())}"
        )

    random_state = details["random_state"]
    min_periods = details["min_periods"]
    span = details["span"]

    art_names = [a.name for a in run.logged_artifacts() if "-history" not in a.name.lower()]
    if not art_names:
        raise ValueError(f"No non-history artifact found for run {run_id}")
    last_artifact_name = art_names[-1]

    model, pipeline = load_model_and_pipeline(
        run_id,
        last_artifact_name,
        wandb_project,
        entity=entity,
    )

    orientation_run = details.get("orientation", orientation)
    bet_type_run = details.get("bet_type", bet_type)
    ctx = get_betting_context(orientation_run, bet_type_run)
    data, pass_rates, win_percentages, schedules = load_data()
    df_prepared = prepare_df(
        data=data,
        pass_rates=pass_rates,
        win_percentages=win_percentages,
        schedules=schedules,
        min_periods=min_periods,
        span=span,
        orientation=orientation_run,
        bet_type=bet_type_run,
    )
    cy_df = df_prepared[df_prepared["season"] == cy_year]
    df_prepared = df_prepared[df_prepared["season"] < cy_year]

    target = ctx["target"]
    model_type = "regression" if bet_type == "spread" else "classification"
    test_size = 0.1
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        df_prepared[features],
        df_prepared[target],
        test_size=test_size,
        stratify=df_prepared[target],
        random_state=random_state,
    )
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_train_df,
        y_train_df,
        test_size=test_size,
        stratify=y_train_df,
        random_state=random_state,
    )

    df_train = df_prepared.loc[X_train_df.index].copy()
    df_val = df_prepared.loc[X_val_df.index].copy()
    df_test = df_prepared.loc[X_test_df.index].copy()

    test_res = evaluate_betting_strategy(
        df=df_test,
        model=model,
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
    cy_res = evaluate_betting_strategy(
        df=cy_df,
        model=model,
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
    return test_res, cy_res


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = [
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


def filter_eval_results(
    df: pd.DataFrame, roi_min: float = 10.0, bet_rate_min: float = 6.0
) -> pd.DataFrame:
    """Filter evaluation results and keep best per run and strategy."""
    roi_cols = ["train_roi", "val_roi", "test_roi"]
    filt = (df[roi_cols] >= roi_min).all(axis=1) & (df["test_bet_rate"] >= bet_rate_min)
    df = df.loc[filt].copy()
    if df.empty:
        return df
    df["avg_roi"] = df[roi_cols].mean(axis=1)
    df = (
        df.sort_values("avg_roi", ascending=False)
        .drop_duplicates(subset=["run_id", "bet_strat"], keep="first")
        .sort_values("avg_roi", ascending=False)
    )
    return df.drop(columns="avg_roi")


def aggregated_roi_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean ROI by strategy and margin."""
    roi_cols = [c for c in ["train_roi", "val_roi", "test_roi", "cy_roi"] if c in df.columns]
    return df.groupby(["bet_strat", "margin"])[roi_cols].mean().round(2)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="W&B evaluation utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    runs_p = sub.add_parser("runs", help="evaluate top runs and show ROI table")
    runs_p.add_argument("--project", required=True, help="W&B project")
    runs_p.add_argument("--top-metric", default="loss", help="metric to rank runs")
    runs_p.add_argument("--top-n", type=int, default=10, help="number of runs")
    runs_p.add_argument("--train-weight", type=float, default=1.0)
    runs_p.add_argument("--metric-threshold", type=float)
    runs_p.add_argument("--exclude-tested", action="store_true")
    runs_p.add_argument("--pull-high-roi", action="store_true")
    runs_p.add_argument(
        "--orientation",
        choices=["fav_dog", "home_away"],
        default="fav_dog",
    )
    runs_p.add_argument(
        "--bet-type",
        choices=["moneyline", "spread"],
        default="moneyline",
        dest="bet_type",
    )

    single_p = sub.add_parser(
        "single",
        help="evaluate one run and write prediction CSVs",
    )
    single_p.add_argument("--project", required=True, help="W&B project")
    single_p.add_argument("--run-id", required=True, help="run id")
    single_p.add_argument("--bet-strat", required=True, help="betting strategy")
    single_p.add_argument("--margin", type=float, required=True)
    single_p.add_argument(
        "--orientation",
        choices=["fav_dog", "home_away"],
        default="fav_dog",
    )
    single_p.add_argument(
        "--bet-type",
        choices=["moneyline", "spread"],
        default="moneyline",
        dest="bet_type",
    )
    single_p.add_argument(
        "--output",
        default=RESULTS_DIR,
        help="directory to save prediction CSVs",
    )

    args = parser.parse_args(argv)
    if args.command == "runs":
        results = run_pipeline(
            wandb_project=args.project,
            features=DEFAULT_FEATURES,
            top_metric=args.top_metric,
            top_n=args.top_n,
            train_weight=args.train_weight,
            exclude_tested=args.exclude_tested,
            pull_high_roi=args.pull_high_roi,
            metric_threshold=args.metric_threshold,
            orientation=args.orientation,
            bet_type=args.bet_type,
        )
        if results.empty:
            print("No results")
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = f"{RESULTS_DIR}/wandb_eval_{args.orientation}_{args.bet_type}.csv"
        results.to_csv(out_path, index=False)
        filtered = filter_eval_results(results)
        if filtered.empty:
            print("No runs met filter criteria")
        else:
            print(filtered.reset_index(drop=True).to_string(index=False))
    elif args.command == "single":
        test_res, cy_res = evaluate_single_run(
            run_id=args.run_id,
            features=DEFAULT_FEATURES,
            bet_strat=args.bet_strat,
            margin=args.margin,
            wandb_project=args.project,
            orientation=args.orientation,
            bet_type=args.bet_type,
        )
        out_dir = args.output or RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        test_path = os.path.join(out_dir, f"{args.run_id}_test_preds.csv")
        test_res["df"].to_csv(test_path, index=False)
        print(f"Test ROI: {test_res['roi']:.2f}% -> {test_path}")
        cy_path = os.path.join(out_dir, f"{args.run_id}_cy_preds.csv")
        cy_res["df"].to_csv(cy_path, index=False)
        print(f"Current year ROI: {cy_res['roi']:.2f}% -> {cy_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
