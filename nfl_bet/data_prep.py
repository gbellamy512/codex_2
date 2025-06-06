import pandas as pd
import numpy as np
from .betting import implied_probability


def build_full_schedule(df, team_col="team", season_weeks=17):
    """Create a complete schedule for each team/week."""
    teams = df[team_col].unique()
    seasons = df["season"].unique()
    idx = pd.MultiIndex.from_product(
        [seasons, range(1, season_weeks + 1), teams],
        names=["season", "week", team_col],
    )
    return idx.to_frame(index=False)


def merge_and_ffill(df, schedule, group_cols, merge_cols, value_col):
    merged = pd.merge(schedule, df, on=merge_cols, how="left")
    merged[value_col] = merged.groupby(group_cols)[value_col].ffill()
    merged["week"] += 1
    return merged


def add_h2h(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["h2h_team"] = None
    for index, row in data.iterrows():
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        week = row["week"]
        games = data[
            (data["season"] == season)
            & (
                ((data["home_team"] == home_team) & (data["away_team"] == away_team))
                | ((data["home_team"] == away_team) & (data["away_team"] == home_team))
            )
            & (data["week"] < week)
        ]
        if not games.empty:
            last_game = games.sort_values(by="week", ascending=False).iloc[0]
            if last_game["home_score"] > last_game["away_score"]:
                data.at[index, "h2h_team"] = last_game["home_team"]
            elif last_game["away_score"] > last_game["home_score"]:
                data.at[index, "h2h_team"] = last_game["away_team"]
            else:
                data.at[index, "h2h_team"] = "tie"
    
    def assign_h2h_type(row):
        if pd.isna(row["h2h_team"]):
            return "na"
        if row["h2h_team"] == "tie":
            return "tie"
        if row["h2h_team"] == row["home_team"]:
            return "home"
        return "away"

    data["h2h_type"] = data.apply(assign_h2h_type, axis=1)
    data.drop(columns=["h2h_team"], inplace=True)
    return data


def determine_favorite(row):
    if row["home_moneyline"] < row["away_moneyline"]:
        return "home", "away"
    if row["away_moneyline"] < row["home_moneyline"]:
        return "away", "home"
    return ("home", "away") if row["home_moneyline"] < 0 else ("away", "home")


def prepare_df(
    data: pd.DataFrame,
    pass_rates: pd.DataFrame,
    win_percentages: pd.DataFrame,
    schedules: pd.DataFrame,
    *,
    min_periods: int = 3,
    span: int = 8,
    avg_method: str = "ewma",
    orientation: str = "fav_dog",
    bet_type: str = "moneyline",
) -> pd.DataFrame:
    df = data.copy()
    metric = (
        f"ewma_{min_periods}min_{span}span" if avg_method == "ewma" else "epa_season"
    )
    keep_cols = [
        c
        for c in df.columns
        if c == "home_team_win"
        or metric in c
        or c in ["season", "week", "home_team", "away_team"]
    ]
    df = df[keep_cols]

    pass_rates_sched = build_full_schedule(pass_rates, "team", season_weeks=17)
    pass_rates_merged = merge_and_ffill(
        pass_rates,
        pass_rates_sched,
        group_cols=["season", "team"],
        merge_cols=["season", "week", "team"],
        value_col="pass_rate_ytd",
    )
    df = pd.merge(
        df,
        pass_rates_merged.rename(
            columns={"team": "home_team", "pass_rate_ytd": "pass_rate_ytd_home"}
        ),
        on=["season", "week", "home_team"],
        how="left",
    )
    df = pd.merge(
        df,
        pass_rates_merged.rename(
            columns={"team": "away_team", "pass_rate_ytd": "pass_rate_ytd_away"}
        ),
        on=["season", "week", "away_team"],
        how="left",
    )

    win_perc_sched = build_full_schedule(win_percentages, "team", season_weeks=17)
    win_perc_merged = merge_and_ffill(
        win_percentages,
        win_perc_sched,
        group_cols=["season", "team"],
        merge_cols=["season", "week", "team"],
        value_col="win_percentage",
    )
    win_perc_merged = win_perc_merged[["season", "week", "team", "win_percentage"]]
    df = pd.merge(
        df,
        win_perc_merged.rename(
            columns={"team": "home_team", "win_percentage": "win_percentage_home"}
        ),
        on=["season", "week", "home_team"],
        how="left",
    )
    df = pd.merge(
        df,
        win_perc_merged.rename(
            columns={"team": "away_team", "win_percentage": "win_percentage_away"}
        ),
        on=["season", "week", "away_team"],
        how="left",
    )

    schedules = schedules[
        [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_moneyline",
            "away_moneyline",
            "spread_line",
            "away_spread_odds",
            "home_spread_odds",
            "result",
            "home_rest",
            "away_rest",
            "div_game",
            "weekday",
        ]
    ].copy()
    schedules["weekday"] = schedules["weekday"].apply(
        lambda x: x if x in ["Monday", "Thursday", "Sunday"] else "other"
    )
    df = pd.merge(
        df,
        schedules,
        on=["season", "week", "home_team", "away_team"],
        how="inner",
    )
    df = add_h2h(df)
    df.dropna(inplace=True)
    df["home_line"] = -df["spread_line"]
    df["away_line"] = df["spread_line"]

    if orientation == "fav_dog":
        df[["favorite", "dog"]] = df.apply(determine_favorite, axis=1, result_type="expand")
        df["dog_home"] = df["favorite"].apply(lambda x: 1 if x == "away" else 0)

        def map_h2h_type(row):
            if row["h2h_type"] == "home":
                return "dog" if row["dog_home"] else "fav"
            if row["h2h_type"] == "away":
                return "fav" if row["dog_home"] else "dog"
            return "na"

        df["h2h_type"] = df.apply(map_h2h_type, axis=1)
        home_cols = [col for col in df.columns if "home" in col]
        away_cols = [col for col in df.columns if "away" in col]
        for home_col in home_cols:
            away_col = home_col.replace("home", "away")
            fav_col = home_col.replace("home", "fav")
            dog_col = home_col.replace("home", "dog")
            if away_col in df.columns:
                df[fav_col] = df.apply(
                    lambda row: row[home_col] if row["favorite"] == "home" else row[away_col],
                    axis=1,
                )
                df[dog_col] = df.apply(
                    lambda row: row[away_col] if row["favorite"] == "home" else row[home_col],
                    axis=1,
                )
        df.drop(columns=(home_cols + away_cols + ["favorite", "dog"]), inplace=True)
        df["dog_win"] = df.apply(lambda row: 1 if row["dog_score"] > row["fav_score"] else 0, axis=1)

        if bet_type == "spread":
            df["dog_margin"] = df["dog_score"] + df["dog_line"] - df["fav_score"]
            df["dog_cover"] = df.apply(
                lambda row: 1 if row["dog_margin"] > 0 else 0,
                axis=1,
            )
            df["fav_implied_prob"] = df["fav_spread_odds"].apply(implied_probability)
            df["dog_implied_prob"] = df["dog_spread_odds"].apply(implied_probability)
        else:
            df["fav_implied_prob"] = df["fav_moneyline"].apply(implied_probability)
            df["dog_implied_prob"] = df["dog_moneyline"].apply(implied_probability)
        df["implied_prob_diff"] = df["dog_implied_prob"] - df["fav_implied_prob"]
        df["sunday"] = df["weekday"].apply(lambda x: 1 if x == "Sunday" else 0)

        if avg_method == "ewma":
            df["rushing_offense_adv"] = df[f"ewma_{min_periods}min_{span}span_rushing_offense_net_dog"] - df[
                f"ewma_{min_periods}min_{span}span_rushing_defense_net_fav"
            ]
            df["passing_offense_adv"] = df[f"ewma_{min_periods}min_{span}span_passing_offense_net_dog"] - df[
                f"ewma_{min_periods}min_{span}span_passing_defense_net_fav"
            ]
            df["rushing_defense_adv"] = df[f"ewma_{min_periods}min_{span}span_rushing_defense_net_dog"] - df[
                f"ewma_{min_periods}min_{span}span_rushing_offense_net_fav"
            ]
            df["passing_defense_adv"] = df[f"ewma_{min_periods}min_{span}span_passing_defense_net_dog"] - df[
                f"ewma_{min_periods}min_{span}span_passing_offense_net_fav"
            ]
        else:
            df["rushing_offense_adv"] = df["rushing_offense_epa_season_dog"] + df[
                "rushing_defense_epa_season_fav"
            ]
            df["passing_offense_adv"] = df["passing_offense_epa_season_dog"] + df[
                "passing_defense_epa_season_fav"
            ]
            df["rushing_defense_adv"] = -df["rushing_defense_epa_season_dog"] - df[
                "rushing_offense_epa_season_fav"
            ]
            df["passing_defense_adv"] = -df["passing_defense_epa_season_dog"] - df[
                "passing_offense_epa_season_fav"
            ]
        df["win_percentage_diff"] = df["win_percentage_dog"] - df["win_percentage_fav"]
        df["rest_advantage"] = df["dog_rest"] - df["fav_rest"]
    else:
        df["home_win"] = df["home_team_win"]
        if bet_type == "spread":
            df["home_margin"] = df["home_score"] + df["home_line"] - df["away_score"]
            df["home_cover"] = df.apply(
                lambda row: 1 if row["home_margin"] > 0 else 0,
                axis=1,
            )
            df["home_implied_prob"] = df["home_spread_odds"].apply(implied_probability)
            df["away_implied_prob"] = df["away_spread_odds"].apply(implied_probability)
        else:
            df["home_implied_prob"] = df["home_moneyline"].apply(implied_probability)
            df["away_implied_prob"] = df["away_moneyline"].apply(implied_probability)
        df["implied_prob_diff"] = df["away_implied_prob"] - df["home_implied_prob"]
        df["sunday"] = df["weekday"].apply(lambda x: 1 if x == "Sunday" else 0)

        if avg_method == "ewma":
            df["rushing_offense_adv"] = df[f"ewma_{min_periods}min_{span}span_rushing_offense_net_away"] - df[
                f"ewma_{min_periods}min_{span}span_rushing_defense_net_home"
            ]
            df["passing_offense_adv"] = df[f"ewma_{min_periods}min_{span}span_passing_offense_net_away"] - df[
                f"ewma_{min_periods}min_{span}span_passing_defense_net_home"
            ]
            df["rushing_defense_adv"] = df[f"ewma_{min_periods}min_{span}span_rushing_defense_net_away"] - df[
                f"ewma_{min_periods}min_{span}span_rushing_offense_net_home"
            ]
            df["passing_defense_adv"] = df[f"ewma_{min_periods}min_{span}span_passing_defense_net_away"] - df[
                f"ewma_{min_periods}min_{span}span_passing_offense_net_home"
            ]
        else:
            df["rushing_offense_adv"] = df["rushing_offense_epa_season_away"] + df[
                "rushing_defense_epa_season_home"
            ]
            df["passing_offense_adv"] = df["passing_offense_epa_season_away"] + df[
                "passing_defense_epa_season_home"
            ]
            df["rushing_defense_adv"] = -df["rushing_defense_epa_season_away"] - df[
                "rushing_offense_epa_season_home"
            ]
            df["passing_defense_adv"] = -df["passing_defense_epa_season_away"] - df[
                "passing_offense_epa_season_home"
            ]
        df["win_percentage_diff"] = df["win_percentage_away"] - df["win_percentage_home"]
        df["rest_advantage"] = df["away_rest"] - df["home_rest"]
    return df
