import argparse
import os
import pandas as pd
from nfl_bet import (
    prepare_df,
    train_model,
    evaluate_betting_strategy,
    get_betting_context,
    filter_results_df,
)
from nfl_bet.constants import DEFAULT_FEATURES

DATA_DIR = "data"
RESULTS_DIR = "results"
FIRST_YEAR = 2013
CURRENT_YEAR = 2024


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/{name}")


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


def run_once(
    orientation: str,
    bet_type: str,
    *,
    data: pd.DataFrame,
    pass_rates: pd.DataFrame,
    win_percentages: pd.DataFrame,
    schedules: pd.DataFrame,
    margin: float = 0.0,
    save_csv: bool = False,
) -> None:
    """Train and evaluate a single orientation/bet type combination."""
    model_type = "regression" if bet_type == "spread" else "classification"

    df = prepare_df(
        data=data,
        pass_rates=pass_rates,
        win_percentages=win_percentages,
        schedules=schedules,
        avg_method="simple",
        orientation=orientation,
        bet_type=bet_type,
    )
    features = DEFAULT_FEATURES
    context = get_betting_context(orientation, bet_type)
    cat_features = ["h2h_type", "div_game"]
    target_col = (
        context["regression_target"] if model_type == "regression" else context["target"]
    )
    model, pipeline, (X_test, y_test) = train_model(
        df,
        features,
        categorical_features=cat_features,
        target=target_col,
        model_type=model_type,
    )
    results = evaluate_betting_strategy(
        df.loc[y_test.index],
        model,
        features=features,
        pipeline=pipeline,
        margin=margin,
        target=context["target"],
        team1_label=context["team1_label"],
        team2_label=context["team2_label"],
        team1_odds_col=context["team1_odds_col"],
        team2_odds_col=context["team2_odds_col"],
        model_type=model_type,
        line_col=context.get("line_col"),
    )
    df_results = results["df"]
    if save_csv:
        df_trimmed = filter_results_df(
            df_results,
            features,
            orientation,
            bet_type,
        )
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = f"{RESULTS_DIR}/betting_results_{orientation}_{bet_type}.csv"
        df_trimmed.to_csv(out_path, index=False)
    print(f"{orientation} {bet_type} ROI: {results['roi']:.2f}%")


def main(argv: None | list[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orientation",
        choices=["fav_dog", "home_away"],
        default="fav_dog",
    )
    parser.add_argument(
        "--bet-type",
        choices=["moneyline", "spread"],
        default="moneyline",
        dest="bet_type",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help=(
            "Save detailed betting results to "
            "results/betting_results_<orientation>_<bet-type>.csv"
        ),
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all orientation/bet-type combinations sequentially",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Minimum edge required to place a bet",
    )
    args = parser.parse_args(argv)
    data, pass_rates, win_percentages, schedules = load_data()

    if args.run_all:
        combos = [
            ("fav_dog", "moneyline"),
            ("fav_dog", "spread"),
            ("home_away", "moneyline"),
            ("home_away", "spread"),
        ]
        for orientation, bet_type in combos:
            run_once(
                orientation,
                bet_type,
                data=data,
                pass_rates=pass_rates,
                win_percentages=win_percentages,
                schedules=schedules,
                margin=args.margin,
                save_csv=True,
            )
    else:
        run_once(
            args.orientation,
            args.bet_type,
            data=data,
            pass_rates=pass_rates,
            win_percentages=win_percentages,
            schedules=schedules,
            margin=args.margin,
            save_csv=args.save_csv,
        )


if __name__ == "__main__":
    main()
