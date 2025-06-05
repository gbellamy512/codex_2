import pandas as pd


def implied_probability(moneyline: float) -> float:
    """Convert American moneyline odds to implied win probability."""
    if moneyline > 0:
        return 100 / (moneyline + 100)
    return -moneyline / (-moneyline + 100)


def calculate_implied_probabilities(df: pd.DataFrame, fav_col: str = "fav_moneyline", dog_col: str = "dog_moneyline") -> pd.DataFrame:
    df = df.copy()
    df["fav_implied_prob"] = df[fav_col].apply(implied_probability)
    df["dog_implied_prob"] = df[dog_col].apply(implied_probability)
    return df


def determine_bet_team(row: pd.Series, margin: float = 0.0) -> str:
    if row["predictions"] > (row["dog_implied_prob"] + margin):
        return "dog"
    if (1 - row["predictions"]) > (row["fav_implied_prob"] + margin):
        return "fav"
    return "none"


def simple_bet_team(row: pd.Series) -> str:
    return "dog" if row["predictions"] > 0.5 else "fav"


def calculate_stake(row: pd.Series, bet_amount: float, bet_strat: str = "both") -> float:
    should_bet = False
    if bet_strat == "both":
        should_bet = row["bet_team"] != "none"
    elif bet_strat == "dog":
        should_bet = row["bet_team"] == "dog"
    elif bet_strat == "fav":
        should_bet = row["bet_team"] == "fav"
    return bet_amount if should_bet else 0.0


def calculate_profit(row: pd.Series) -> float:
    if row["bet_team"] == "fav":
        if row["dog_win"] == 0:
            return row["bet"] * (100 / -row["fav_moneyline"] if row["fav_moneyline"] < 0 else row["fav_moneyline"] / 100)
        return -row["bet"]
    if row["bet_team"] == "dog":
        if row["dog_win"] == 1:
            return row["bet"] * (100 / -row["dog_moneyline"] if row["dog_moneyline"] < 0 else row["dog_moneyline"] / 100)
        return -row["bet"]
    return 0.0


def calculate_dog_fav_profits(df: pd.DataFrame, bet_amount: float = 100) -> pd.DataFrame:
    df = df.copy()
    df["dog_profit"] = df.apply(lambda r: bet_amount * (r["dog_moneyline"] / 100) if r["dog_win"] == 1 else -bet_amount, axis=1)
    df["fav_profit"] = df.apply(lambda r: bet_amount * (100 / abs(r["fav_moneyline"])) if r["dog_win"] == 0 else -bet_amount, axis=1)
    return df


def evaluate_betting_strategy(
    df: pd.DataFrame,
    model,
    features,
    pipeline,
    bet_amount: float = 100,
    target: str = "dog_win",
    use_implied_prob: bool = True,
    bet_strat: str = "both",
    margin: float = 0.0,
) -> dict:
    X = df[features]
    X_norm = pipeline.transform(X)
    if hasattr(model, "predict_proba"):
        predictions = model.predict_proba(X_norm)[:, 1]
    else:
        predictions = model.predict(X_norm)
    df_eval = df.copy()
    df_eval["predictions"] = predictions
    df_eval = calculate_implied_probabilities(df_eval)
    if use_implied_prob:
        df_eval["bet_team"] = df_eval.apply(determine_bet_team, margin=margin, axis=1)
    else:
        df_eval["bet_team"] = df_eval.apply(simple_bet_team, axis=1)
    df_eval["bet"] = df_eval.apply(lambda r: calculate_stake(r, bet_amount, bet_strat), axis=1)
    df_eval["profit"] = df_eval.apply(calculate_profit, axis=1)
    total_profit = df_eval["profit"].sum()
    total_bet = df_eval["bet"].sum()
    roi = (total_profit / total_bet * 100) if total_bet else 0.0
    df_eval = calculate_dog_fav_profits(df_eval, bet_amount=bet_amount)
    dog_win_rate = df_eval[df_eval[target] == 1].shape[0] / len(df_eval) if len(df_eval) else 0.0
    return {
        "total_profit": total_profit,
        "total_bet": total_bet,
        "roi": roi,
        "dog_win_rate": dog_win_rate,
        "df": df_eval,
        "bet_rate": 100 * len(df_eval[df_eval["bet"] > 0]) / len(df_eval),
    }
