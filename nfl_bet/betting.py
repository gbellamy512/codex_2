"""Betting utilities.

The :func:`evaluate_betting_strategy` helper returns a DataFrame containing a
large number of columns. Use :func:`filter_results_df` to trim that DataFrame to
the key information typically saved in result CSVs.
"""

import pandas as pd


def implied_probability(moneyline: float) -> float:
    """Convert American moneyline odds to implied win probability."""
    if moneyline > 0:
        return 100 / (moneyline + 100)
    return -moneyline / (-moneyline + 100)


def calculate_implied_probabilities(
    df: pd.DataFrame,
    *,
    team1_odds_col: str,
    team2_odds_col: str,
    team1_label: str = "dog",
    team2_label: str = "fav",
) -> pd.DataFrame:
    """Add implied probability columns for the provided odds columns."""
    df = df.copy()
    df[f"{team1_label}_implied_prob"] = df[team1_odds_col].apply(implied_probability)
    df[f"{team2_label}_implied_prob"] = df[team2_odds_col].apply(implied_probability)
    return df


def determine_bet_team(
    row: pd.Series,
    *,
    margin: float = 0.0,
    team1_label: str = "dog",
    team2_label: str = "fav",
    team1_prob_col: str,
    team2_prob_col: str,
) -> str:
    """Return which side to bet based on predictions and implied probabilities."""
    if row["predictions"] > (row[team1_prob_col] + margin):
        return team1_label
    if (1 - row["predictions"]) > (row[team2_prob_col] + margin):
        return team2_label
    return "none"


def simple_bet_team(
    row: pd.Series, *, team1_label: str = "dog", team2_label: str = "fav"
) -> str:
    """Bet on the side with the higher predicted probability."""
    return team1_label if row["predictions"] > 0.5 else team2_label


def determine_spread_bet_team(
    row: pd.Series,
    *,
    line_col: str,
    threshold: float = 0.0,
    team1_label: str = "dog",
    team2_label: str = "fav",
) -> str:
    """Return which side to bet based on predicted margin vs. the spread."""
    predicted = row["predictions"]
    line = row[line_col]
    if predicted > line + threshold:
        return team1_label
    if predicted < line - threshold:
        return team2_label
    return "none"


def calculate_stake(
    row: pd.Series,
    bet_amount: float,
    bet_strat: str = "both",
    *,
    team1_label: str = "dog",
    team2_label: str = "fav",
) -> float:
    """Return the stake amount for a given strategy."""
    should_bet = False
    if bet_strat == "both":
        should_bet = row["bet_team"] != "none"
    elif bet_strat == team1_label:
        should_bet = row["bet_team"] == team1_label
    elif bet_strat == team2_label:
        should_bet = row["bet_team"] == team2_label
    return bet_amount if should_bet else 0.0


def calculate_profit(
    row: pd.Series,
    *,
    team1_label: str = "dog",
    team2_label: str = "fav",
    team1_odds_col: str,
    team2_odds_col: str,
    outcome_col: str,
) -> float:
    """Compute profit for a single game based on the bet outcome."""
    if row["bet_team"] == team2_label:
        if row[outcome_col] == 0:
            odds = row[team2_odds_col]
            return row["bet"] * (100 / -odds if odds < 0 else odds / 100)
        return -row["bet"]
    if row["bet_team"] == team1_label:
        if row[outcome_col] == 1:
            odds = row[team1_odds_col]
            return row["bet"] * (100 / -odds if odds < 0 else odds / 100)
        return -row["bet"]
    return 0.0


def calculate_fixed_side_profits(
    df: pd.DataFrame,
    *,
    team1_odds_col: str,
    team2_odds_col: str,
    outcome_col: str,
    team1_label: str = "dog",
    team2_label: str = "fav",
    bet_amount: float = 100,
) -> pd.DataFrame:
    """Compute profits if always betting on a given side."""
    df = df.copy()

    def payout(odds: float) -> float:
        return bet_amount * (100 / -odds) if odds < 0 else bet_amount * (odds / 100)

    def check_push(row: pd.Series) -> bool:
        margin_cols = [f"{team1_label}_margin", f"{team2_label}_margin"]
        for col in margin_cols:
            if col in row.index and row[col] == 0:
                return True
        if {"home_score", "away_score"}.issubset(row.index):
            if row["home_score"] == row["away_score"]:
                return True
        return False

    def team1_profit(row: pd.Series) -> float:
        if check_push(row):
            return 0.0
        if row[outcome_col] == 1:
            return payout(row[team1_odds_col])
        return -bet_amount

    def team2_profit(row: pd.Series) -> float:
        if check_push(row):
            return 0.0
        if row[outcome_col] == 0:
            return payout(row[team2_odds_col])
        return -bet_amount

    df[f"{team1_label}_profit"] = df.apply(team1_profit, axis=1)
    df[f"{team2_label}_profit"] = df.apply(team2_profit, axis=1)
    return df


def get_betting_context(orientation: str, bet_type: str) -> dict:
    """Return context information for the specified orientation and bet type.

    The dictionary always includes classification targets and odds columns.
    When ``bet_type`` is ``spread`` additional keys ``line_col`` and
    ``regression_target`` identify the spread line and the margin column used for
    regression style models.
    """
    orientation = orientation.lower()
    bet_type = bet_type.lower()
    if orientation not in {"fav_dog", "home_away"}:
        raise ValueError(f"Invalid orientation: {orientation}")
    if bet_type not in {"moneyline", "spread"}:
        raise ValueError(f"Invalid bet_type: {bet_type}")

    line_col = None
    regression_target = None

    if orientation == "fav_dog":
        if bet_type == "moneyline":
            team1_label = "dog"
            team2_label = "fav"
            target = "dog_win"
            team1_odds_col = "dog_moneyline"
            team2_odds_col = "fav_moneyline"
        else:
            team1_label = "fav"
            team2_label = "dog"
            target = "fav_cover"
            regression_target = "fav_margin"
            line_col = "fav_line"
            team1_odds_col = "fav_spread_odds"
            team2_odds_col = "dog_spread_odds"
    else:
        team1_label = "home"
        team2_label = "away"
        if bet_type == "moneyline":
            target = "home_win"
            team1_odds_col = "home_moneyline"
            team2_odds_col = "away_moneyline"
        else:
            target = "home_cover"
            regression_target = "home_margin"
            line_col = "home_line"
            team1_odds_col = "home_spread_odds"
            team2_odds_col = "away_spread_odds"

    return {
        "target": target,
        "team1_label": team1_label,
        "team2_label": team2_label,
        "team1_odds_col": team1_odds_col,
        "team2_odds_col": team2_odds_col,
        "line_col": line_col,
        "regression_target": regression_target,
    }


def evaluate_betting_strategy(
    df: pd.DataFrame,
    model,
    features,
    pipeline,
    *,
    bet_amount: float = 100,
    target: str = "dog_win",
    team1_label: str = "dog",
    team2_label: str = "fav",
    team1_odds_col: str = "dog_moneyline",
    team2_odds_col: str = "fav_moneyline",
    use_implied_prob: bool = True,
    bet_strat: str = "both",
    margin: float = 0.0,
    model_type: str = "classification",
    line_col: str | None = None,
) -> dict:
    X = df[features]
    X_norm = pipeline.transform(X)
    if model_type == "regression":
        predictions = model.predict(X_norm)
    elif hasattr(model, "predict_proba"):
        predictions = model.predict_proba(X_norm)[:, 1]
    else:
        predictions = model.predict(X_norm)
    df_eval = df.copy()
    df_eval["predictions"] = predictions
    df_eval = calculate_implied_probabilities(
        df_eval,
        team1_odds_col=team1_odds_col,
        team2_odds_col=team2_odds_col,
        team1_label=team1_label,
        team2_label=team2_label,
    )
    if model_type == "regression":
        if line_col is None:
            raise ValueError("line_col must be provided for regression models")
        df_eval["bet_team"] = df_eval.apply(
            determine_spread_bet_team,
            line_col=line_col,
            threshold=margin,
            team1_label=team1_label,
            team2_label=team2_label,
            axis=1,
        )
    else:
        if use_implied_prob:
            df_eval["bet_team"] = df_eval.apply(
                determine_bet_team,
                margin=margin,
                team1_label=team1_label,
                team2_label=team2_label,
                team1_prob_col=f"{team1_label}_implied_prob",
                team2_prob_col=f"{team2_label}_implied_prob",
                axis=1,
            )
        else:
            df_eval["bet_team"] = df_eval.apply(
                simple_bet_team,
                team1_label=team1_label,
                team2_label=team2_label,
                axis=1,
            )
    df_eval["bet"] = df_eval.apply(
        lambda r: calculate_stake(
            r,
            bet_amount,
            bet_strat,
            team1_label=team1_label,
            team2_label=team2_label,
        ),
        axis=1,
    )
    df_eval["profit"] = df_eval.apply(
        calculate_profit,
        team1_label=team1_label,
        team2_label=team2_label,
        team1_odds_col=team1_odds_col,
        team2_odds_col=team2_odds_col,
        outcome_col=target,
        axis=1,
    )
    total_profit = df_eval["profit"].sum()
    total_bet = df_eval["bet"].sum()
    roi = (total_profit / total_bet * 100) if total_bet else 0.0
    df_eval = calculate_fixed_side_profits(
        df_eval,
        team1_odds_col=team1_odds_col,
        team2_odds_col=team2_odds_col,
        outcome_col=target,
        team1_label=team1_label,
        team2_label=team2_label,
        bet_amount=bet_amount,
    )
    win_rate = (
        df_eval[df_eval[target] == 1].shape[0] / len(df_eval) if len(df_eval) else 0.0
    )
    return {
        "total_profit": total_profit,
        "total_bet": total_bet,
        "roi": roi,
        "win_rate": win_rate,
        "df": df_eval,
        "bet_rate": 100 * len(df_eval[df_eval["bet"] > 0]) / len(df_eval),
    }


def filter_results_df(df: pd.DataFrame, features: list[str], orientation: str, bet_type: str) -> pd.DataFrame:
    """Return a trimmed version of the betting results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned in the ``df`` key of :func:`evaluate_betting_strategy`.
    features : list[str]
        List of feature column names used for modeling.
    orientation : str
        Either ``"fav_dog"`` or ``"home_away"``.
    bet_type : str
        ``"moneyline"`` or ``"spread"``.
    """
    orientation = orientation.lower()
    bet_type = bet_type.lower()
    if orientation not in {"fav_dog", "home_away"}:
        raise ValueError(f"Invalid orientation: {orientation}")
    if bet_type not in {"moneyline", "spread"}:
        raise ValueError(f"Invalid bet_type: {bet_type}")

    if orientation == "fav_dog":
        team1, team2 = "fav", "dog"
        result_col = "dog_win" if bet_type == "moneyline" else "fav_cover"
    else:
        team1, team2 = "home", "away"
        result_col = "home_win" if bet_type == "moneyline" else "home_cover"

    cols = [
        "season",
        "week",
        f"{team1}_team",
        f"{team2}_team",
        f"{team1}_score",
        f"{team2}_score",
        result_col,
    ]

    if bet_type == "moneyline":
        cols += [f"{team1}_moneyline", f"{team2}_moneyline"]
    else:
        cols += ["spread_line", f"{team1}_spread_odds", f"{team2}_spread_odds"]

    cols += list(features) + ["bet_team", "bet", "profit"]
    return df[[c for c in cols if c in df.columns]].copy()
