import argparse
import pandas as pd
from nfl_bet import (
    prepare_df,
    train_model,
    evaluate_betting_strategy,
    get_betting_context,
)

DATA_DIR = "data"
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
        help="Save detailed betting results to betting_results.csv",
    )
    args = parser.parse_args(argv)

    model_type = "regression" if args.bet_type == "spread" else "classification"

    data, pass_rates, win_percentages, schedules = load_data()
    df = prepare_df(
        data=data,
        pass_rates=pass_rates,
        win_percentages=win_percentages,
        schedules=schedules,
        avg_method="simple",
        orientation=args.orientation,
        bet_type=args.bet_type,
    )
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
    context = get_betting_context(args.orientation, args.bet_type)
    cat_features = ["h2h_type", "div_game"]
    target_col = context["regression_target"] if model_type == "regression" else context["target"]
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
        target=context["target"],
        team1_label=context["team1_label"],
        team2_label=context["team2_label"],
        team1_odds_col=context["team1_odds_col"],
        team2_odds_col=context["team2_odds_col"],
        model_type=model_type,
        line_col=context.get("line_col"),
    )
    df_results = results["df"]
    if args.save_csv:
        df_results.to_csv("betting_results.csv", index=False)
    print("ROI: {:.2f}%".format(results["roi"]))


if __name__ == "__main__":
    main()
