#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title load libraries and mount colab
# set seed for reproducibility
seed = 1

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf

import random

from joblib import load

# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb

import pprint
from sklearn.utils.class_weight import compute_class_weight
import joblib


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
wandb.login()



# In[ ]:


# @title load data
# ----------------------------------------------------------------
# 1. Load Data (outside of the function; do this once at notebook start)
# ----------------------------------------------------------------

first_year = 2013
current_year = 2024
include_last_week = True

# Historical + Current Year EPA data
data_hist = pd.read_csv(
    f'data/ewma_epa_df_2006_{current_year-1}.csv'
)
data_cy = pd.read_csv(
    f'data/ewma_epa_df_{current_year}.csv'
)
data = pd.concat([data_hist, data_cy])

if include_last_week:
  data = data.loc[
      ((data['season'] < 2022) & (data['week'] <= 17))
      | ((data['season'] >= 2022) & (data['week'] <= 18))
  ]
else:
  data = data.loc[
    ((data['season'] < 2022) & (data['week'] <= 16))
    | ((data['season'] >= 2022) & (data['week'] <= 17))
]

# Pass Rate data
pass_rates_hist = pd.read_csv(
    f'data/pass_rates_{first_year}_{current_year-1}.csv'
)
pass_rates_cy = pd.read_csv(
    f'data/pass_rates_{current_year}.csv'
)
pass_rates_hist_ancient = pd.read_csv(
    f'data/pass_rates_2006_2012.csv'
)
pass_rates = pd.concat([
    pass_rates_hist_ancient, pass_rates_hist, pass_rates_cy
])

# Win percentage data
win_percentages_hist = pd.read_csv(
    f'data/weekly_records_1999_{current_year-1}.csv'
)
win_percentages_cy = pd.read_csv(
    f'data/weekly_records_{current_year}.csv'
)
win_percentages = pd.concat([win_percentages_hist, win_percentages_cy])

# Schedules data
schedules_hist = pd.read_csv(
    f'data/nfl_schedules_1999_{current_year-1}.csv'
)
schedules_cy = pd.read_csv(
    f'data/nfl_schedules_{current_year}.csv'
)
schedules = pd.concat([schedules_hist, schedules_cy])

# ----------------------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------------------

def build_full_schedule(df, team_col='team', season_weeks=17):
    """
    Creates a complete schedule of season x [1..season_weeks] x teams.
    This helps to fill missing rows for bye weeks, etc.
    """
    teams = df[team_col].unique()
    seasons = df['season'].unique()
    # Create an index for all (season, week, team) combinations
    schedule_idx = pd.MultiIndex.from_product(
        [seasons, range(1, season_weeks + 1), teams],
        names=['season', 'week', team_col]
    )
    return schedule_idx.to_frame(index=False)

def merge_and_ffill(df, schedule, group_cols, merge_cols, value_col):
    """
    Merges `df` with a complete `schedule` to fill missing weeks/rows,
    then forward-fills the specified `value_col`.
    After that, increments the 'week' by 1 (so it applies to the next game).
    """
    merged = pd.merge(schedule, df, on=merge_cols, how='left')
    # Forward-fill for each (season, team)
    merged[value_col] = merged.groupby(group_cols)[value_col].ffill()
    # Shift week by 1 so it applies to the next matchup
    merged['week'] += 1
    return merged

def add_h2h(data):
    """
    For each row, look up the most recent head-to-head matchup
    and determine which team won. Adds 'h2h_type' to the dataframe:
      - 'home' if last time the home team won
      - 'away' if last time the away team won
      - 'tie' if last time was a tie
      - 'na' if there was no last time (no previous matchup)
    """
    data = data.copy()
    data['h2h_team'] = None

    for index, row in data.iterrows():
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']
        week = row['week']

        # Filter: same season, same matchup, prior week
        relevant_games = data[
            (data['season'] == season)
            & (
                ((data['home_team'] == home_team) & (data['away_team'] == away_team)) |
                ((data['home_team'] == away_team) & (data['away_team'] == home_team))
            )
            & (data['week'] < week)
        ]

        if not relevant_games.empty:
            last_game = relevant_games.sort_values(by='week', ascending=False).iloc[0]

            if last_game['home_score'] > last_game['away_score']:
                data.at[index, 'h2h_team'] = last_game['home_team']
            elif last_game['away_score'] > last_game['home_score']:
                data.at[index, 'h2h_team'] = last_game['away_team']
            else:
                data.at[index, 'h2h_team'] = "tie"

    # Map the winner to 'home', 'away', or 'tie'
    def assign_h2h_type(row):
        if pd.isna(row['h2h_team']):
            return 'na'
        elif row['h2h_team'] == 'tie':
            return 'tie'
        elif row['h2h_team'] == row['home_team']:
            return 'home'
        else:
            return 'away'

    data['h2h_type'] = data.apply(assign_h2h_type, axis=1)
    data.drop(columns=['h2h_team'], inplace=True)
    return data

def determine_favorite(row):
    """
    Determine which team is favored based on moneyline.
      - If home_moneyline < away_moneyline -> home is favorite, away is dog
      - If away_moneyline < home_moneyline -> away is favorite, home is dog
      - If equal, break ties by sign convention (negative means favorite).
    """
    if row['home_moneyline'] < row['away_moneyline']:
        return 'home', 'away'
    elif row['away_moneyline'] < row['home_moneyline']:
        return 'away', 'home'
    else:
        # Tie moneylines: fallback
        return ('home', 'away') if row['home_moneyline'] < 0 else ('away', 'home')

def implied_probability(moneyline):
    """
    Converts American moneyline odds to implied probability.
    """
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)

# ----------------------------------------------------------------
# 3. Main Function: prepare_df
# ----------------------------------------------------------------

def prepare_df(
    min_periods=3,
    span=8,
    avg_method='ewma',  # new parameter to choose averaging method
    data=data.copy(),
    pass_rates=pass_rates.copy(),
    win_percentages=win_percentages.copy(),
    schedules=schedules.copy()
):
    """
    Prepare a single dataframe that merges:
      - Weighted EPA data (already loaded + filtered globally)
      - Pass rates (forward-filled to handle bye weeks)
      - Win percentages (forward-filled)
      - Additional scheduling, moneyline, rest data
      - Creates columns for favorite/dog, h2h outcome, rest advantage, etc.

    The avg_method parameter can be either 'ewma' or 'simple'.
    'ewma' (default) uses exponentially weighted moving averages,
    while 'simple' uses the season-level simple averages (e.g., columns like rushing_offense_epa_season).
    """
    df = data.copy()

    # -------------- COLUMN SELECTION --------------
    # Choose the metric substring based on the averaging method.
    if avg_method == 'ewma':
        metric = f'ewma_{min_periods}min_{span}span'
    elif avg_method == 'simple':
        metric = 'epa_season'
    else:
        raise ValueError("avg_method must be either 'ewma' or 'simple'")

    # Keep only columns for modeling and 'home_team_win'
    keep_cols = [
        c for c in df.columns
        if c == 'home_team_win'
        or metric in c
        or c in ['season', 'week', 'home_team', 'away_team']
    ]
    df = df[keep_cols]

    # -------------- PASS RATES --------------
    pass_rates_sched = build_full_schedule(pass_rates, 'team', season_weeks=17)
    pass_rates_merged = merge_and_ffill(
        pass_rates,
        pass_rates_sched,
        group_cols=['season', 'team'],
        merge_cols=['season', 'week', 'team'],
        value_col='pass_rate_ytd'
    )
    # Merge pass_rate_ytd for home and away teams
    df = pd.merge(
        df,
        pass_rates_merged.rename(columns={
            'team': 'home_team',
            'pass_rate_ytd': 'pass_rate_ytd_home'
        }),
        on=['season', 'week', 'home_team'],
        how='left'
    )
    df = pd.merge(
        df,
        pass_rates_merged.rename(columns={
            'team': 'away_team',
            'pass_rate_ytd': 'pass_rate_ytd_away'
        }),
        on=['season', 'week', 'away_team'],
        how='left'
    )

    # -------------- WIN PERCENTAGES --------------
    win_perc_sched = build_full_schedule(win_percentages, 'team', season_weeks=17)
    win_perc_merged = merge_and_ffill(
        win_percentages,
        win_perc_sched,
        group_cols=['season', 'team'],
        merge_cols=['season', 'week', 'team'],
        value_col='win_percentage'
    )
    win_perc_merged = win_perc_merged[['season', 'week', 'team', 'win_percentage']]

    # Merge win_percentage for home and away teams
    df = pd.merge(
        df,
        win_perc_merged.rename(columns={
            'team': 'home_team',
            'win_percentage': 'win_percentage_home'
        }),
        on=['season', 'week', 'home_team'],
        how='left'
    )
    df = pd.merge(
        df,
        win_perc_merged.rename(columns={
            'team': 'away_team',
            'win_percentage': 'win_percentage_away'
        }),
        on=['season', 'week', 'away_team'],
        how='left'
    )

    # -------------- SCHEDULES / MONEYLINES / REST --------------
    schedules = schedules[['season', 'week', 'home_team', 'away_team',
                            'home_score', 'away_score',
                            'home_moneyline', 'away_moneyline',
                            'home_rest', 'away_rest', 'div_game', 'weekday']].copy()
    schedules['weekday'] = schedules['weekday'].apply(
        lambda x: x if x in ['Monday', 'Thursday', 'Sunday'] else 'other'
    )
    df = pd.merge(
        df,
        schedules,
        on=['season', 'week', 'home_team', 'away_team'],
        how='inner'
    )

    # -------------- HEAD-TO-HEAD COLUMN --------------
    df = add_h2h(df)

    # Drop rows that have missing data after merges
    df.dropna(inplace=True)

    # -------------- FAVORITE / DOG IDENTIFICATION --------------
    df[['favorite', 'dog']] = df.apply(determine_favorite, axis=1, result_type="expand")
    df['dog_home'] = df['favorite'].apply(lambda x: 1 if x == 'away' else 0)

    def map_h2h_type(row):
        if row['h2h_type'] == 'home':
            return 'dog' if row['dog_home'] else 'fav'
        elif row['h2h_type'] == 'away':
            return 'fav' if row['dog_home'] else 'dog'
        else:
            return 'na'

    df['h2h_type'] = df.apply(map_h2h_type, axis=1)

    # -------------- FAVORITE / DOG COLUMNS --------------
    home_cols = [col for col in df.columns if 'home' in col]
    away_cols = [col for col in df.columns if 'away' in col]

    for home_col in home_cols:
        away_col = home_col.replace('home', 'away')
        fav_col  = home_col.replace('home', 'fav')
        dog_col  = home_col.replace('home', 'dog')

        if away_col in df.columns:
            df[fav_col] = df.apply(
                lambda row: row[home_col] if row['favorite'] == 'home' else row[away_col],
                axis=1
            )
            df[dog_col] = df.apply(
                lambda row: row[away_col] if row['favorite'] == 'home' else row[home_col],
                axis=1
            )

    df.drop(columns=(home_cols + away_cols + ['favorite', 'dog']), inplace=True)

    # -------------- TARGET: DID THE DOG WIN? --------------
    df['dog_win'] = df.apply(lambda row: 1 if row['dog_score'] > row['fav_score'] else 0, axis=1)

    # -------------- IMPLIED PROBABILITIES --------------
    df['fav_implied_prob'] = df['fav_moneyline'].apply(implied_probability)
    df['dog_implied_prob'] = df['dog_moneyline'].apply(implied_probability)

    # -------------- WEEKDAY INDICATOR --------------
    df['sunday'] = df['weekday'].apply(lambda x: 1 if x == 'Sunday' else 0)

    # -------------- ADVANTAGE COLUMNS --------------
    if avg_method == 'ewma':
        # Use the existing exponentially weighted moving average columns
        df['rushing_offense_adv'] = (
            df[f'ewma_{min_periods}min_{span}span_rushing_offense_net_dog']
            - df[f'ewma_{min_periods}min_{span}span_rushing_defense_net_fav']
        )
        df['passing_offense_adv'] = (
            df[f'ewma_{min_periods}min_{span}span_passing_offense_net_dog']
            - df[f'ewma_{min_periods}min_{span}span_passing_defense_net_fav']
        )
        df['rushing_defense_adv'] = (
            df[f'ewma_{min_periods}min_{span}span_rushing_defense_net_dog']
            - df[f'ewma_{min_periods}min_{span}span_rushing_offense_net_fav']
        )
        df['passing_defense_adv'] = (
            df[f'ewma_{min_periods}min_{span}span_passing_defense_net_dog']
            - df[f'ewma_{min_periods}min_{span}span_passing_offense_net_fav']
        )
    else:
        df['rushing_offense_adv'] = (
            df['rushing_offense_epa_season_dog']
            + df['rushing_defense_epa_season_fav']
        )
        df['passing_offense_adv'] = (
            df['passing_offense_epa_season_dog']
            + df['passing_defense_epa_season_fav']
        )
        df['rushing_defense_adv'] = (
            -1 * df['rushing_defense_epa_season_dog']
            - df['rushing_offense_epa_season_fav']
        )
        df['passing_defense_adv'] = (
            -1 * df['passing_defense_epa_season_dog']
            - df['passing_offense_epa_season_fav']
        )

    # -------------- OTHER DERIVED FEATURES --------------
    df['win_percentage_diff'] = df['win_percentage_dog'] - df['win_percentage_fav']
    df['implied_prob_diff']   = df['dog_implied_prob'] - df['fav_implied_prob']
    df['rest_advantage']      = df['dog_rest'] - df['fav_rest']

    return df

# ----------------------------------------------------------------
# 4. Run the function
# ----------------------------------------------------------------

# avg_method = 'ewma'
avg_method = 'simple'
df = prepare_df(avg_method=avg_method)
features = [
    f'rushing_offense_adv',
    f'passing_offense_adv',
    f'rushing_defense_adv',
    f'passing_defense_adv',
    'win_percentage_diff',
    'implied_prob_diff',
    'rest_advantage',
    'div_game',
    'h2h_type',
]


categorical_features  = ['h2h_type']
binary_features = ['div_game']
non_numerical_features = binary_features + categorical_features
numerical_features = [feature for feature in features if feature not in non_numerical_features]

cy = 2024
cy_df = df[df['season'] == cy]
df = df[df['season'] < cy]

print('before current year')
print(len(df))
print(df['dog_win'].value_counts(normalize=True))
print('')
print('current year')
print(len(cy_df))
print(cy_df['dog_win'].value_counts(normalize=True))

# In[ ]:


from re import VERBOSE
# @title create betting logic
def implied_probability(moneyline):
    """
    Convert American moneyline odds to implied probability.
    """
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return -moneyline / (-moneyline + 100)

def calculate_implied_probabilities(df, fav_col='fav_moneyline', dog_col='dog_moneyline'):
    """
    Append columns with implied probabilities for the favorite and underdog.
    """
    df['fav_implied_prob'] = df[fav_col].apply(implied_probability)
    df['dog_implied_prob'] = df[dog_col].apply(implied_probability)
    return df

def determine_bet_team(row, margin=0.0):
    """
    Logic to decide which team to bet on based on:
      - model prediction
      - implied probabilities
      - margin
    Returns 'dog', 'fav', or 'none'.
    """
    # If the dog’s predicted win probability exceeds dog_implied_prob + margin
    # we bet on the dog.
    if row['predictions'] > (row['dog_implied_prob'] + margin):
        return 'dog'
    # If the favorite’s predicted win probability exceeds fav_implied_prob + margin
    # i.e. (1 - row['predictions']) > (row['fav_implied_prob'] + margin)
    elif (1 - row['predictions']) > (row['fav_implied_prob'] + margin):
        return 'fav'
    else:
        return 'none'

def simple_bet_team(row):
    """
    If the dog's predicted win probability is > 0.5, bet on the dog;
    otherwise, bet on the favorite.
    """
    return 'dog' if row['predictions'] > 0.5 else 'fav'

def calculate_stake(row, bet_amount, bet_strat='both'):
    """
    Determine the stake amount (0 or bet_amount) depending on bet_strat:
      - 'both': bet on whichever team bet_team says (if bet_team != 'none')
      - 'dog': only bet if bet_team == 'dog'
      - 'fav': only bet if bet_team == 'fav'
    """
    should_bet = False

    if bet_strat == 'both':
        should_bet = (row['bet_team'] != 'none')
    elif bet_strat == 'dog':
        should_bet = (row['bet_team'] == 'dog')
    elif bet_strat == 'fav':
        should_bet = (row['bet_team'] == 'fav')

    return bet_amount if should_bet else 0

def calculate_profit(row):
    """
    Calculate profit (or loss) for a single row, given:
      - which team was bet on
      - who actually won (dog_win)
      - moneyline for each team
      - bet amount
    """
    # If we bet on the favorite
    if row['bet_team'] == 'fav':
        if row['dog_win'] == 0:  # Favorite won
            # For negative moneylines, payout is bet * (100 / -moneyline)
            # For positive moneylines, payout is bet * (moneyline / 100)
            return (
                row['bet'] * (100 / -row['fav_moneyline'])
                if row['fav_moneyline'] < 0
                else row['bet'] * (row['fav_moneyline'] / 100)
            )
        else:
            return -row['bet']

    # If we bet on the dog
    elif row['bet_team'] == 'dog':
        if row['dog_win'] == 1:  # Dog won
            return (
                row['bet'] * (100 / -row['dog_moneyline'])
                if row['dog_moneyline'] < 0
                else row['bet'] * (row['dog_moneyline'] / 100)
            )
        else:
            return -row['bet']

    # No bet placed
    return 0

def calculate_dog_fav_profits(df, fav_moneyline_col='fav_moneyline',
                              dog_moneyline_col='dog_moneyline',
                              dog_win_col='dog_win', bet_amount=100):
    """
    Calculate hypothetical profits if a user always bets on dog or
    always bets on fav for each game. (For reference/comparison.)
    """
    # Profit if you always bet on the dog
    df['dog_profit'] = df.apply(
        lambda row: bet_amount * (row[dog_moneyline_col] / 100)
        if row[dog_win_col] == 1 else -bet_amount,
        axis=1
    )

    # Profit if you always bet on the favorite
    df['fav_profit'] = df.apply(
        lambda row: bet_amount * (100 / abs(row[fav_moneyline_col]))
        if row[dog_win_col] == 0 else -bet_amount,
        axis=1
    )

    return df

def evaluate_betting_strategy(df,
                              model,
                              features,
                              pipeline,
                              bet_amount=100,
                              target='dog_win',
                              use_implied_prob=True,
                              bet_strat='both',
                              margin=0.0,
                              # suppresses the bars
                              keras_verbose=0):

    # 1. Prepare input features
    X = df[features]

    X_norm = pipeline.transform(X)

    # 2. Predict dog’s win probability
    # predictions = model.predict(X_norm)
    predictions = model.predict(X_norm, verbose=keras_verbose)
    df_eval = df.copy()  # Work off a copy to avoid modifying original DataFrame
    df_eval['predictions'] = predictions

    # 3. Calculate implied probabilities
    df_eval = calculate_implied_probabilities(df_eval)

    # 4. Determine which team to bet on
    if use_implied_prob:
        df_eval['bet_team'] = df_eval.apply(
            determine_bet_team, margin=margin, axis=1
        )
    else:
        df_eval['bet_team'] = df_eval.apply(simple_bet_team, axis=1)

    # 5. Decide bet amount based on bet_strat
    df_eval['bet'] = df_eval.apply(
        lambda row: calculate_stake(row, bet_amount, bet_strat),
        axis=1
    )

    # 6. Calculate profit or loss for each game
    df_eval['profit'] = df_eval.apply(calculate_profit, axis=1)

    # 7. Summaries of overall profit / ROI
    total_profit = df_eval['profit'].sum()
    total_bet = df_eval['bet'].sum()
    return_on_investment = (total_profit / total_bet * 100) if total_bet else 0.0

    # 8. Calculate hypothetical “always dog” or “always fav” returns
    df_eval = calculate_dog_fav_profits(df_eval,
                                        fav_moneyline_col='fav_moneyline',
                                        dog_moneyline_col='dog_moneyline',
                                        dog_win_col=target,
                                        bet_amount=bet_amount)

    total_bet_everything = len(df_eval) * bet_amount
    fav_profit = df_eval['fav_profit'].sum()
    fav_return_on_investment = (fav_profit / total_bet_everything * 100)

    dog_profit = df_eval['dog_profit'].sum()
    dog_return_on_investment = (dog_profit / total_bet_everything * 100)

    # 9. Dog win rate
    dog_win_rate = (df_eval[df_eval[target] == 1].shape[0] / len(df_eval)
                    if len(df_eval) > 0 else 0.0)

    # 10. Compile results into a dictionary
    results = {
        "total_profit": total_profit,
        "total_bet": total_bet,
        "roi": return_on_investment,
        # "fav_roi": fav_return_on_investment,
        # "dog_roi": dog_return_on_investment,
        "dog_win_rate": dog_win_rate,
        # Return the full evaluation DataFrame in case you need details
        "df": df_eval,
        'bet_rate': 100 * len(df_eval[df_eval['bet'] > 0]) / len(df_eval)
    }

    return results

# gemini
# 
# I would highly recommend starting with:
# 
# Optimizer: adam
# Activation: relu
# Initializer: he_normal

# In[ ]:


# @title run sweep

wandb_project = 'nfl_bet_sweep_2'
project_name = wandb_project

def create_sweep():

  sweep_config = {
      # 'method': 'random'
      'method': 'bayes'
  }

  metric = {
      # 'name': 'test_loss',
      'name': 'val_loss',
      'goal': 'minimize'
  }

  sweep_config['metric'] = metric

  parameters_dict = {
      # 'optimizer': {
      #     'values': ['adam', 'sgd', 'rmsprop']
      # },
      # 'sgd_momentum': {
      #     'values': [0.0, 0.5, 0.9]
      # },
      # 'kernel_initializer': {
      #     'values': [
      #         'he_normal',
      #         'glorot_normal',
      #     ]
      # },
      # 'activation': {
      #     'values': [
      #         'relu',
      #         'elu',
      #     ]
      # },
      # 'learning_rate': {
      #     'values': [0.01, 0.001, 0.0005, 0.0001]
      # },
      'learning_rate': {
          'distribution': 'log_uniform_values',
          'min': 1e-5,
          'max': 1e-2
      },
      # 'epochs': {
      #     # 'values': [25, 50, 75, 100]
      #     'values': [200]
      # },
      'batch_size': {
          'values': [16, 32, 64]
      },
      # 'hidden_layers': {
      #     'values': [2, 3, 4, 5]
      # },
      'hidden_layers': {
          'distribution': 'int_uniform',
          'min': 2,
          'max': 4
      },
      'neurons': {
          'values': [16, 32, 64]
      },
      'batch_norm': {
          'values': [True, False]
      },
      # 'l1_reg': {
      #     'values': [0.0, 0.001, 0.01, 0.1]
      # },
      # 'l2_reg': {
      #     'values': [0.0, 0.001, 0.01, 0.1]
      # },
      # 'dropout': {
      #     'values': [0.0, 0.1, 0.2, 0.3]
      # },
      'l1_reg': {
          'distribution': 'log_uniform_values',
          'min': 1e-6,
          'max': 1e-2
      },
      'l2_reg': {
          'distribution': 'log_uniform_values',
          'min': 1e-6,
          'max': 1e-2
      },
      'dropout': {
          'distribution': 'uniform',
          'min': 0.0,
          'max': 0.5
      },
      'apply_class_weights': {
          'values': [True, False]
      },
      # 'min_periods': {
      #     'values': [3, 6]
      # },
      # 'span': {
      #     'values': [3, 8, 16]
      # },
      # 'early_stopping': {
      #     # 'values': [True, False]
      #     'values': [True]
      # },
      # 'early_stopping_patience': {
      #     'values': [10, 20, 30]
      # },
      'early_stopping_patience': {
          'distribution': 'int_uniform',
          'min': 10,  # Or a bit lower if you want to explore shorter patience
          'max': 35   # Or a bit higher if you want to explore longer patience
      },
      # 'early_stopping_min_delta': {
      #     'values': [0.001, 0.0001, 0.00001]
      # },
      'early_stopping_min_delta': {
          'distribution': 'log_uniform_values',
          'min': 1e-5,  # log(0.00001)
          'max': 1e-3   # log(0.001)
      },
      # 'apply_lr_schedule': {
      #     # 'values': [True, False]
      #     'values': [True]
      # },
      # 'lr_scheduler_step_every': {
      #     'values': [5, 10, 25]
      # },
      'lr_scheduler_step_every': {
          'distribution': 'int_uniform',
          'min': 5,
          'max': 30 # Or adjust based on your typical total epochs
      },
      # 'lr_scheduler_step_factor': {
      #     'values': [0.25, 0.5, 0.75]
      # }
      'lr_scheduler_step_factor': {
          'distribution': 'uniform', # or 'log_uniform'
          'min': 0.1,  # More aggressive decay
          'max': 0.8   # More gentle decay
      }
  }

  sweep_config['parameters'] = parameters_dict

  gpu_devices = tf.config.list_physical_devices('GPU')
  gpu_name = gpu_devices[0].name if gpu_devices else 'CPU'

  # Add fixed-value parameters
  parameters_dict.update({
      'gpu_name': {
          'value': gpu_name
      },
      'target': {
          'value': 'dog_win'
      },
      'loss': {
          'value': 'binary_crossentropy'
      },
      'metric': {
          'value': 'accuracy'
      },
     'optimizer': {
          'value': 'adam'
      },
      'kernel_initializer': {
          'value': 'he_normal'
      },
      'activation': {
          'value': 'relu'
      },
      'outer_activation': {
          'value': 'sigmoid'
      },
      'epochs': {
          'value': 200
      },
      'early_stopping': {
          'value': True
      },
      'apply_lr_schedule': {
          'value': True
      },
      'test_size': {
          'value': 0.1
      },
      'min_periods': {
          'value': 3
      },
      'span': {
          'value': 8
      },
      # 'bet_strat': {
      #     'value': 'both'
      # },
      # 'margin': {
      #     'value': 0.00
      # },
  })

  pprint.pprint(sweep_config)

  sweep_id = wandb.sweep(sweep_config, project=project_name)

  return sweep_id

precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

def evaluate_and_log_metrics(model, X, y, tag):
    # Evaluate the model on the provided data
    loss, accuracy, precision, recall = model.evaluate(X, y, verbose=0)

    # Calculate F1 Score based on precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    # Log metrics with the given tag (train, val, or test)
    wandb.log({
        f"{tag}_loss": loss,
        f"{tag}_accuracy": accuracy,
        f"{tag}_precision": precision,
        f"{tag}_recall": recall,
        f"{tag}_f1": f1_score,
        f"{tag}_len": len(X)
    })

def evaluate_and_log_metrics_betting(dataset, prefix, model, pipeline, features, bet_strat, margin):
    results = evaluate_betting_strategy(dataset, model, pipeline=pipeline, features=features,
                                        bet_strat=bet_strat, margin=margin)
    wandb.log({
        f"{prefix}_profit": results["total_profit"],
        f"{prefix}_bet": results["total_bet"],
        f"{prefix}_roi": results["roi"],
    })

def get_activation(activation_name):
    """Helper function to handle activation functions"""
    if activation_name == 'elu':
        return tf.keras.activations.elu
    return activation_name  # 'relu' and others work directly

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        keras_verbose = 0
        save_freq_epochs = 10

        # Prepare your dataframe for training
        df = prepare_df(min_periods=config.min_periods, span=config.span)

        cy = 2024
        cy_df = df[df['season'] == cy]
        df = df[df['season'] < cy]

        # ColumnTransformer to handle numerical and categorical preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features + binary_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        # Generate or define a random seed
        random_state = random.randint(0, 1_000)
        wandb.log({"random_state": random_state})

        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            df[features],
            df[config.target],
            test_size=config.test_size,
            stratify=df[config.target],
            random_state=random_state
        )
        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            X_train_df,
            y_train_df,
            test_size=config.test_size,
            stratify=y_train_df,
            random_state=random_state
        )

        # Apply the transformations to the datasets
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        X_train = pipeline.fit_transform(X_train_df)
        X_val = pipeline.transform(X_val_df)
        X_test = pipeline.transform(X_test_df)

        pipeline_file = "pipeline.pkl"
        joblib.dump(pipeline, pipeline_file)
        artifact = wandb.Artifact(f'preprocessing_pipeline_{wandb.run.id}', type='pipeline')
        artifact.add_file(pipeline_file)
        wandb.log_artifact(artifact)

        df_train = df.loc[X_train_df.index].copy()
        df_val = df.loc[X_val_df.index].copy()
        df_test = df.loc[X_test_df.index].copy()

        y_train = y_train_df.values
        y_val = y_val_df.values
        y_test = y_test_df.values

        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
        for i in range(config.hidden_layers):
            if config.batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config.dropout))
            model.add(tf.keras.layers.Dense(
                config.neurons,
                activation=get_activation(config.activation),
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1_reg, l2=config.l2_reg)
            ))
        model.add(tf.keras.layers.Dropout(config.dropout))
        model.add(tf.keras.layers.Dense(1, activation=config.outer_activation))

        # Choose optimizer
        if config.optimizer == 'sgd':
            momentum_value = float(config.sgd_momentum)
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=config.learning_rate,
                momentum=momentum_value
            )
        elif config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        elif config.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=config.loss,
            metrics=[config.metric, precision_metric, recall_metric]
        )

        wandb_callbacks = [WandbMetricsLogger()]

        # 2) Early Stopping (if enabled)
        if config.early_stopping:
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                restore_best_weights=True
            )
            wandb_callbacks.append(early_stopping_cb)

        # 3) Learning Rate Scheduler (if enabled)
        if config.apply_lr_schedule:
            # # Example: reduce LR by factor of 10 after 10 epochs
            # def scheduler(epoch, lr):
            #     # After 10 epochs, reduce LR by a factor of 10
            #     return lr * 0.1 if epoch >= 10 else lr
            def scheduler(epoch, lr):
                """Reduce the LR by `factor` every `step_every` epochs."""
                if (epoch + 1) % config.lr_scheduler_step_every == 0:
                    return lr * config.lr_scheduler_step_factor
                return lr

            lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
            wandb_callbacks.append(lr_scheduler_cb)

        # Compute class weights if flagged
        if config.apply_class_weights:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None

        # Train the model
        model.fit(
            x=X_train,
            y=y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=wandb_callbacks,
            verbose=keras_verbose,
            class_weight=class_weight_dict
        )

        # Save the model
        model_file = "model.keras"
        model.save(model_file)
        artifact = wandb.Artifact(f'model_{wandb.run.id}', type='model')
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)

        # Evaluate and log metrics on train/val/test
        evaluate_and_log_metrics(model, X_train, y_train, "train")
        evaluate_and_log_metrics(model, X_val, y_val, "val")
        evaluate_and_log_metrics(model, X_test, y_test, "test")

        # Prepare the cy_df data
        X_cy = cy_df[features]
        y_cy = cy_df[config.target]
        X_cy = pipeline.transform(X_cy)
        y_cy = y_cy.values

        evaluate_and_log_metrics(model, X_cy, y_cy, "cy")

        # # Evaluate betting metrics
        # evaluate_and_log_metrics_betting(cy_df, "cy", model, pipeline, features, config.bet_strat, config.margin)
        # evaluate_and_log_metrics_betting(df_train, "train", model, pipeline, features, config.bet_strat, config.margin)
        # evaluate_and_log_metrics_betting(df_val, "val", model, pipeline, features, config.bet_strat, config.margin)
        # evaluate_and_log_metrics_betting(df_test, "test", model, pipeline, features, config.bet_strat, config.margin)


# sweep_id = create_sweep()
# print(sweep_id)
# wandb.agent(sweep_id, train, count=50)
wandb.agent(sweep_id='grantbell/nfl_bet_sweep_2/qm2m2u47', function=train, count=50)

# In[ ]:


# @title get and test wandb models
"""
Module for evaluating NFL betting strategies using W&B run artifacts.

This module contains helper functions to:
  - Retrieve the top N runs from a W&B project based on a specified performance metric.
  - Download and load model and preprocessing pipeline artifacts.
  - Evaluate betting performance (profit, bet, ROI) on train, validation, test, and optional current-year data.
  - Execute evaluations across multiple runs and betting strategies.
  - Evaluate a single run's model performance on different dataset splits.

High-Level Flow Diagram:
----------------------------------------------------
run_pipeline()
   ├── get_top_runs_quickly()
   └── exe()
         ├── load_model_and_pipeline()
         ├── prepare_df()  <-- external function
         └── evaluate_betting_results()
----------------------------------------------------
"""


# =============================================================================
# Main Pipeline Entry Points
# =============================================================================

# def run_pipeline(project_path, features, top_metric='loss', top_n=10,
#                  train_weight=1.0, wandb_project='nfl_bet_sweep_8',
#                  bet_strats=None, margins=None, cy_df=None,
#                  exclude_tested: bool = True):
def run_pipeline(wandb_project, features, top_metric='loss', top_n=10,
                 train_weight=1.0,
                 bet_strats=None, margins=None, cy_df=None,
                 exclude_tested: bool = True,
                 pull_high_roi: bool = False,
                 metric_threshold=0.60):
    """
    Convenience function to run the entire evaluation pipeline:
      1. Retrieve the top N runs from W&B based on the specified metric.
      2. Execute betting evaluation across different strategies and margins.

    Returns:
        pd.DataFrame: Combined results from all runs and evaluation settings.
    """
    # Step 1: Retrieve top runs from W&B
    top_runs_df = get_top_runs_quickly(
        # project_path=project_path,
        wandb_project=wandb_project,
        metric=top_metric,
        top_n=top_n,
        train_weight=train_weight,
        metric_threshold=metric_threshold
    )

    # # Save the top runs information to CSV for record keeping
    # top_runs_df.to_csv('top_runs.csv', index=False)

    # Step 2: Execute the betting evaluation for the selected runs
    results_df = exe(
        df_runs_epochs=top_runs_df,
        # base_df=base_df,
        features=features,
        wandb_project=wandb_project,
        bet_strats=bet_strats,
        margins=margins,
        cy_df=cy_df
        , exclude_tested=exclude_tested
        , pull_high_roi=pull_high_roi
    )

    return results_df


# =============================================================================
# Utility Functions
# =============================================================================

def get_top_runs_quickly(
    # project_path='nfl_bet_sweep_8',
    wandb_project,
    # Random guessing baseline: around 0.693 (because log(0.5)=0.693−log(0.5)=0.693). If your positive class is only 33%, then a random guess loss would be closer to .636.
    metric_threshold,
    metric='loss',
    # top_n=10,
    top_n=None,
    train_weight=1.0
):
    """
    Retrieve and process only the top N runs for a project based on a specified metric,
    but first filter out any runs whose aggregate metric doesn't meet the threshold.

    Supported metrics: 'loss', 'accuracy', 'precision', 'recall', 'f1'.
    For 'loss', lower values are better. For the other metrics, higher values are better.

    Args:
        project_path (str): W&B project path, e.g. "user/project".
        metric (str): which metric to rank on.
        top_n (int): how many runs to return.
        train_weight (float): weight to give the training metric.
        metric_threshold (float): per-split cutoff; actual cutoff on aggregate is
                                  metric_threshold * (train_weight + 1 + 1).

    Returns:
        pd.DataFrame of the top runs (after thresholding), with full summary + config.
    """
    # Validate metric choice
    allowed_metrics = {'loss', 'accuracy', 'precision', 'recall', 'f1'}
    if metric not in allowed_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Choose from {allowed_metrics}.")

    lowest_is_better = (metric == 'loss')

    api = wandb.Api()
    # runs = api.runs(project_path)
    runs = api.runs(wandb_project)

    # gather per-run train/val/test
    run_summaries = []
    for run in runs:
        s = run.summary._json_dict
        run_summaries.append({
            'run_id': run.id,
            'run_name': run.name,
            f'train_{metric}': s.get(f'train_{metric}'),
            f'val_{metric}':   s.get(f'val_{metric}'),
            f'test_{metric}':  s.get(f'test_{metric}')
        })

    df = pd.DataFrame(run_summaries).dropna(
        subset=[f'train_{metric}', f'val_{metric}', f'test_{metric}']
    )

    # compute aggregate
    df['aggregate_metric'] = (
        train_weight * df[f'train_{metric}'] +
        df[f'val_{metric}'] +
        df[f'test_{metric}']
    )

    df = df.sort_values(by='aggregate_metric', ascending=lowest_is_better)

    # calculate total threshold
    total_weight = train_weight + 2  # train + val + test
    cutoff = metric_threshold * total_weight

    # pre-filter by threshold
    if lowest_is_better:
        df = df[df['aggregate_metric'] <= cutoff]
    else:
        df = df[df['aggregate_metric'] >= cutoff]

    print(f"Filtered to {len(df)} runs.")

    if top_n:
      # then pick top_n
      if lowest_is_better:
          df = df.nsmallest(top_n, 'aggregate_metric')
      else:
          df = df.nlargest(top_n, 'aggregate_metric')

    # fetch full details for each
    details = []
    for _, row in df.iterrows():
        # run = api.run(f"{project_path}/{row['run_id']}")
        run = api.run(f"{wandb_project}/{row['run_id']}")
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        artifacts = [a.name for a in run.logged_artifacts() if '-history' not in a.name]
        details.append({
            **summary,
            **config,
            'run_id': run.id,
            'run_name': run.name,
            'last_artifact_name': artifacts[-1] if artifacts else None
        })

    return pd.DataFrame(details)

def load_model_and_pipeline(run_id, last_artifact_name, wandb_project):
    """
    Download and load a model and its corresponding preprocessing pipeline from W&B artifacts.

    Returns:
        tuple: (model, pipeline) where model is a TensorFlow Keras model and pipeline is a pre-processing pipeline.
    """
    api = wandb.Api()

    # Download and load the model artifact
    model_artifact = api.artifact(f'grantbell/{wandb_project}/{last_artifact_name}')
    artifact_dir_model = model_artifact.download()
    model = tf.keras.models.load_model(f"{artifact_dir_model}/model.keras")

    # Download and load the preprocessing pipeline artifact (assumes a .pkl file)
    pipeline_artifact = api.artifact(f'grantbell/{wandb_project}/preprocessing_pipeline_{run_id}:v0')
    artifact_dir_pipeline = pipeline_artifact.download()
    pipeline_path = f"{artifact_dir_pipeline}/pipeline.pkl"
    pipeline = load(pipeline_path)

    # Optional: print artifact name for logging
    print(last_artifact_name)

    return model, pipeline


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_betting_results(model, pipeline, features, df_train, df_val, df_test, bet_strat, margin, cy_df=None):
    """
    Evaluate the betting strategy using the provided model and pipeline on train, validation, test,
    and optionally current-year data.

    Returns:
        dict: Aggregated metrics including profit, total bet, and ROI for each dataset.
    """
    # Evaluate betting performance on the training set
    train_res = evaluate_betting_strategy(
        df=df_train,
        model=model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin
    )
    train_metrics = {
        "train_profit": train_res["total_profit"],
        "train_bet":    train_res["total_bet"],
        "train_roi":    train_res["roi"],
        "train_bet_rate": train_res["bet_rate"]
    }

    # Evaluate on the validation set
    val_res = evaluate_betting_strategy(
        df=df_val,
        model=model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin
    )
    val_metrics = {
        "val_profit": val_res["total_profit"],
        "val_bet":    val_res["total_bet"],
        "val_roi":    val_res["roi"],
        "val_bet_rate": val_res["bet_rate"]
    }

    # Evaluate on the test set
    test_res = evaluate_betting_strategy(
        df=df_test,
        model=model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin
    )
    test_metrics = {
        "test_profit": test_res["total_profit"],
        "test_bet":    test_res["total_bet"],
        "test_roi":    test_res["roi"],
        "test_bet_rate": test_res["bet_rate"]
    }

    # If current-year data is provided, evaluate it as well
    cy_metrics = {}
    if cy_df is not None:
        cy_res = evaluate_betting_strategy(
            df=cy_df,
            model=model,
            features=features,
            pipeline=pipeline,
            bet_strat=bet_strat,
            margin=margin
        )
        cy_metrics = {
            "cy_profit": cy_res["total_profit"],
            "cy_bet":    cy_res["total_bet"],
            "cy_roi":    cy_res["roi"],
            "cy_bet_rate": cy_res["bet_rate"]
        }

    # Combine and return all metrics
    return {**train_metrics, **val_metrics, **test_metrics, **cy_metrics}

import re # Import regular expressions for tag parsing

def exe(df_runs_epochs, features, wandb_project='nfl_bet_sweep_8',
        bet_strats=None, margins=None, cy_df=None,
        exclude_tested: bool = True,
        pull_high_roi: bool = False):
    """
    Iterate over runs, load models/pipelines, prepare data, and evaluate betting strategies.

    If pull_high_roi is True, only evaluates strategy/margin combinations found
    in the run's 'high_roi_S_<strat>_M_<margin>' tags. Otherwise, evaluates all
    combinations provided in bet_strats and margins.

    Returns:
        pd.DataFrame: Combined results from all evaluated runs, strategies, and margins.
    """
    # Set default betting strategies and margins if none provided
    # These defaults are primarily used when pull_high_roi is False
    if bet_strats is None:
        bet_strats = ['both', 'dog', 'fav']
    if margins is None:
        margins = [0.025, 0.05, 0.075, 0.1]

    results_list = []
    api = wandb.Api() # Initialize API once

    # Iterate over each run in the provided DataFrame
    for _, row in df_runs_epochs.iterrows():
        run_name = row['run_name']
        run_id = row['run_id']
        last_artifact_name = row['last_artifact_name']

        print(f"\nProcessing run '{run_name}' (ID: {run_id})...")

        # --- Fetch Run Object (Optimized: Fetch only if needed) ---
        run_obj = None
        if exclude_tested or pull_high_roi:
            try:
                run_obj = api.run(f"{wandb_project}/{run_id}")
            except wandb.errors.CommError as e:
                print(f"Error fetching run {run_id} from project {wandb_project}. Skipping. Error: {e}")
                continue

        # --- Filtering Logic ---
        # 1. Skip if already tested (if exclude_tested is True)
        if exclude_tested and run_obj and 'tested' in run_obj.tags:
            print(f"Skipping already tested run '{run_name}' (ID: {run_id}).")
            continue

        # 2. Skip if artifact name contains "history"
        if "history" in str(last_artifact_name).lower():
            print(f"Skipping run '{run_name}' with artifact '{last_artifact_name}' (contains 'history').")
            continue

        # 3. Check for 'high_roi*' tags if pull_high_roi is True
        high_roi_combos_to_evaluate = set() # Use a set to store unique (strat, margin) pairs
        if pull_high_roi:
            if not run_obj: # Should have been fetched above, but double check
                 print(f"Error: Could not fetch run object for {run_id} when pull_high_roi=True. Skipping.")
                 continue

            has_any_high_roi_tag = False
            # Regex to capture strategy and margin from tags like 'high_roi_S_fav_M_50'
            tag_pattern = re.compile(r"high_roi_S_(\w+)_M_(\d+)")

            for tag in run_obj.tags:
                match = tag_pattern.match(tag)
                if match:
                    has_any_high_roi_tag = True
                    strategy = match.group(1)
                    margin_value = int(match.group(2))
                    margin_float = margin_value / 1000.0 # Convert back to float (e.g., 50 -> 0.05)
                    high_roi_combos_to_evaluate.add((strategy, margin_float))
                    print(f"  Found high ROI combo from tag '{tag}': Strategy='{strategy}', Margin={margin_float}")

            if not has_any_high_roi_tag:
                print(f"Skipping run '{run_name}' (ID: {run_id}) as pull_high_roi=True and no 'high_roi_S_*_M_*' tags were found.")
                continue
            elif not high_roi_combos_to_evaluate:
                 print(f"Skipping run '{run_name}' (ID: {run_id}) as no valid combos could be parsed from tags.")
                 continue


        # --- Load Model and Prepare Data (only if not skipped) ---
        try:
            model, pipeline = load_model_and_pipeline(run_id, last_artifact_name, wandb_project)
        except Exception as e:
             print(f"Error loading model/pipeline for run {run_id}. Skipping. Error: {e}")
             continue


        random_state = row['random_state']
        min_periods = row['min_periods']
        span = row['span']
        print(f"  Run Params: random_state={random_state}, min_periods={min_periods}, span={span}")

        # Prepare the dataset
        df_prepared = prepare_df(min_periods=min_periods, span=span)
        cy = 2024 # Consider making this dynamic or a parameter
        # current_cy_df = df_prepared[df_prepared['season'] == cy].copy() if cy_df is None else cy_df.copy()
        current_cy_df = df_prepared[df_prepared['season'] == cy].copy()
        df_prepared = df_prepared[df_prepared['season'] < cy]

        # Split the prepared dataset
        target = 'dog_win'
        test_size = 0.1
        if df_prepared.empty or len(df_prepared[target].unique()) < 2:
             print(f"Skipping run {run_id} due to insufficient data or classes for splitting after filtering.")
             continue

        X = df_prepared[features]
        y = df_prepared[target]

        # Ensure enough samples for stratification
        if y.value_counts().min() < 2:
             print(f"Skipping run {run_id}: Not enough samples in the smallest class for stratification.")
             continue

        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Ensure enough samples for the second split
        if y_train_df.value_counts().min() < 2:
             print(f"Skipping run {run_id}: Not enough samples in the smallest class for validation split stratification.")
             continue

        X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
            X_train_df, y_train_df, test_size=test_size, stratify=y_train_df, random_state=random_state
        )

        df_train = df_prepared.loc[X_train_df.index].copy()
        df_val = df_prepared.loc[X_val_df.index].copy()
        df_test = df_prepared.loc[X_test_df.index].copy()


        # --- Evaluation Loop ---
        run_evaluated = False
        if pull_high_roi:
            # Evaluate ONLY the combinations extracted from tags
            print(f"  Evaluating {len(high_roi_combos_to_evaluate)} high ROI combination(s) found in tags...")
            for bet_strat, margin in high_roi_combos_to_evaluate:
                 print(f"    Evaluating: Strategy='{bet_strat}', Margin={margin}")
                 try:
                     eval_dict = evaluate_betting_results(
                         model=model, pipeline=pipeline, features=features,
                         df_train=df_train, df_val=df_val, df_test=df_test,
                         bet_strat=bet_strat, margin=margin, cy_df=current_cy_df
                     )
                     eval_dict.update({
                         'bet_strat': bet_strat, 'margin': margin, 'run_id': run_id,
                         'last_artifact_name': last_artifact_name, 'run_name': run_name
                     })
                     results_list.append(eval_dict)
                     run_evaluated = True # Mark that at least one evaluation was done for this run
                 except Exception as e:
                     print(f"    Error during evaluation for combo (Strat: {bet_strat}, Margin: {margin}): {e}")
        else:
            # Evaluate ALL combinations from the provided lists (original behavior)
            print(f"  Evaluating all {len(bet_strats)} strategies and {len(margins)} margins...")
            for bet_strat in bet_strats:
                for margin in margins:
                    print(f"    Evaluating: Strategy='{bet_strat}', Margin={margin}")
                    try:
                        eval_dict = evaluate_betting_results(
                            model=model, pipeline=pipeline, features=features,
                            df_train=df_train, df_val=df_val, df_test=df_test,
                            bet_strat=bet_strat, margin=margin, cy_df=current_cy_df
                        )
                        eval_dict.update({
                            'bet_strat': bet_strat, 'margin': margin, 'run_id': run_id,
                            'last_artifact_name': last_artifact_name, 'run_name': run_name
                        })

                        # --- Tagging Logic (Only when NOT pulling high ROI) ---
                        train_roi = eval_dict.get('train_roi', -float('inf'))
                        test_roi = eval_dict.get('test_roi', -float('inf'))
                        val_roi = eval_dict.get('val_roi', -float('inf'))

                        if train_roi > 7.5 and test_roi > 7.5 and val_roi > 7.5:
                            if not run_obj: # Fetch run_obj if not already done
                                try:
                                     run_obj = api.run(f"{wandb_project}/{run_id}")
                                except wandb.errors.CommError as e:
                                     print(f"    Error fetching run {run_id} for tagging. Tagging skipped. Error: {e}")
                                     continue # Skip tagging for this combo

                            if run_obj: # Check if fetch was successful
                                try:
                                    margin_tag_value = int(margin * 1000)
                                    tag_name = f"high_roi_S_{bet_strat}_M_{margin_tag_value}"

                                    if tag_name not in run_obj.tags:
                                        run_obj.tags.append(tag_name)
                                        run_obj.update() # Persist tag change
                                        print(f"    Tagged run '{run_name}' with '{tag_name}'.")
                                    # else: # Optional: print if tag already exists
                                    #     print(f"    Tag '{tag_name}' already exists on run '{run_name}'.")
                                except Exception as e:
                                    print(f"    Error adding tag '{tag_name}' to run '{run_name}': {e}")

                        results_list.append(eval_dict)
                        run_evaluated = True # Mark that at least one evaluation was done
                    except Exception as e:
                        print(f"    Error during evaluation for combo (Strat: {bet_strat}, Margin: {margin}): {e}")


        # --- Add 'tested' Tag (if any evaluation was performed for this run) ---
        if run_evaluated and run_obj and 'tested' not in run_obj.tags:
             try:
                 print(f"  Marking run '{run_name}' as 'tested'.")
                 run_obj.tags.append('tested')
                 run_obj.update()
             except Exception as e:
                 print(f"  Error adding 'tested' tag to run '{run_name}': {e}")
        elif run_evaluated and not run_obj:
             print(f"  Warning: Run {run_id} was evaluated but could not be marked as 'tested' (run object not available).")


    # Combine all results into a single DataFrame
    if not results_list:
        print("\nNo results were generated.")
        return pd.DataFrame() # Return empty DataFrame if no results
    else:
        print(f"\nFinished processing. Combining {len(results_list)} results.")
        return pd.DataFrame(results_list)


def evaluate_single_run(
    # project_path: str,
    run_id: str,
    features: list[str],
    bet_strat: str,
    margin: float,
    # wandb_project: str = 'nfl_bet_sweep_8',
    wandb_project: str,
    cy_year: int = 2024,
) -> pd.DataFrame:
    """
    Evaluate one W&B run with one betting strategy + margin.
    Raises KeyError if 'random_state', 'min_periods', or 'span' are missing
    (no defaults allowed).
    """
    api = wandb.Api()
    # run = api.run(f"{project_path}/{run_id}")
    run = api.run(f"grantbell/{wandb_project}/{run_id}")

    # 1) grab the flat summary dict
    summary = run.summary._json_dict

    # 2) grab config, filtering out private keys
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    # 3) merge exactly as get_top_runs_quickly does
    details = {**summary, **config}

    # 4) enforce no-default requirement
    required = ['random_state', 'min_periods', 'span']
    missing = [k for k in required if k not in details]
    if missing:
        raise KeyError(
            f"Run '{run_id}' missing required key(s): {missing}. "
            f"Available in summary+config: {list(details.keys())}"
        )

    random_state = details['random_state']
    min_periods  = details['min_periods']
    span         = details['span']

    # 5) pick last non-history artifact
    artifact_names = [
        art.name for art in run.logged_artifacts()
        if '-history' not in art.name.lower()
    ]
    if not artifact_names:
        raise ValueError(f"No non-history artifact found for run {run_id}")
    last_artifact_name = artifact_names[-1]

    # 6) load model & pipeline
    model, pipeline = load_model_and_pipeline(run_id, last_artifact_name, wandb_project)

    # Prepare the dataset using an external function (assumed to exist)
    df_prepared = prepare_df(min_periods=min_periods, span=span)
    cy = 2024
    cy_df = df_prepared[df_prepared['season'] == cy]
    df_prepared = df_prepared[df_prepared['season'] < cy]

    # Split the prepared dataset into train, test, and validation sets using stratification
    target = 'dog_win'
    test_size = 0.1

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        df_prepared[features],
        df_prepared[target],
        test_size=test_size,
        stratify=df_prepared[target],
        random_state=random_state
    )
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_train_df,
        y_train_df,
        test_size=test_size,
        stratify=y_train_df,
        random_state=random_state
    )

    # Subset the corresponding rows from the prepared DataFrame
    df_train = df_prepared.loc[X_train_df.index].copy()
    df_val   = df_prepared.loc[X_val_df.index].copy()
    df_test  = df_prepared.loc[X_test_df.index].copy()

    test_res = evaluate_betting_strategy(
        df=df_test,
        model=model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin
    )

    cy_res = evaluate_betting_strategy(
        df=cy_df,
        model=model,
        features=features,
        pipeline=pipeline,
        bet_strat=bet_strat,
        margin=margin
    )

    return test_res, cy_res

from typing import List, Optional

def filter_results_df(
    df: pd.DataFrame,
    train_roi_min: float = 0,
    val_test_roi_min: float = 0,
    cy_roi_min: Optional[float] = None,
    train_bet_min: Optional[float] = None,
    val_bet_min: Optional[float] = None,
    test_bet_min: Optional[float] = None,
    cy_bet_min: Optional[float] = None,
    test_bet_rate_min: Optional[float] = None,
    cols: List[str] = [
        'run_name', 'run_id',
        'train_profit', 'val_profit', 'test_profit', 'cy_profit',
        'train_roi',    'val_roi',    'test_roi',    'cy_roi',
        'train_bet_rate', 'val_bet_rate', 'test_bet_rate', 'cy_bet_rate',
        'cy_bet', 'bet_strat', 'margin'
    ]
) -> pd.DataFrame:
    """
    Filter by ROI and bet thresholds, then select only the columns in `cols`.

    Args:
      df:               your runs DataFrame
      train_roi_min:    minimum train_roi
      val_test_roi_min: minimum val_roi and test_roi
      cy_roi_min:       optional minimum current-year ROI
      train_bet_min:    optional minimum train_bet
      val_bet_min:      optional minimum val_bet
      test_bet_min:     optional minimum test_bet
      cy_bet_min:       optional minimum cy_bet
      cols:             list of columns to return (default as above)

    Returns:
      A filtered DataFrame with only the requested columns.
    """
    # build row-filter mask
    mask = (
        (df['train_roi'] >= train_roi_min) &
        (df['val_roi']   >= val_test_roi_min) &
        (df['test_roi']  >= val_test_roi_min)
    )

    if cy_roi_min is not None:
        mask &= (df['cy_roi'] >= cy_roi_min)
    if train_bet_min is not None:
        mask &= (df['train_bet'] >= train_bet_min)
    if val_bet_min is not None:
        mask &= (df['val_bet'] >= val_bet_min)
    if test_bet_min is not None:
        mask &= (df['test_bet'] >= test_bet_min)
    if cy_bet_min is not None:
        mask &= (df['cy_bet'] >= cy_bet_min)
    if test_bet_rate_min is not None:
        mask &= (df['test_bet_rate'] >= test_bet_rate_min)

    # subset rows and then columns
    return df.loc[mask, cols].copy()

from typing import Dict, Optional

def select_best_by_avg_roi(
    df: pd.DataFrame,
    min_train_bet_rate: Optional[Dict[str, float]] = None,
    min_test_bet_rate: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    df = df.copy()

    # --- 0. apply bet-rate filters if provided ---
    if min_train_bet_rate is not None:
        # map each row's strategy to its threshold, default to -inf so
        # unspecified strategies pass
        train_thresh = df['bet_strat'].map(min_train_bet_rate).fillna(-np.inf)
        df = df.loc[df['train_bet_rate'] >= train_thresh]

    if min_test_bet_rate is not None:
        test_thresh = df['bet_strat'].map(min_test_bet_rate).fillna(-np.inf)
        df = df.loc[df['test_bet_rate']  >= test_thresh]

    # --- 1. Compute per‐row average ROI ---
    roi_cols = ['train_roi', 'val_roi', 'test_roi']
    df['avg_roi'] = df[roi_cols].mean(axis=1)

    # --- 2. Pick best row per run_id × bet_strat ---
    best_idx = df.groupby(['run_id', 'bet_strat'])['avg_roi'].idxmax()

    df.sort_values(by='avg_roi', ascending=False, inplace=True)

    # --- 3. Slice, clean up, and return ---
    result = (
        df
        .loc[best_idx]
        .drop(columns='avg_roi')
        .reset_index(drop=True)
    )
    return result

# In[ ]:


results_df = run_pipeline(
    wandb_project='nfl_bet_sweep_2',
    features=features,
    top_metric='loss',
    top_n=None,
    bet_strats=['dog','fav'],
    margins=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1],
    cy_df=cy_df
    , metric_threshold=.60
    , exclude_tested = True
    # , exclude_tested = False
    # , pull_high_roi = True
)

filtered_df = filter_results_df(results_df, train_roi_min=7.5, val_test_roi_min=7.5)

# Show unique run_id values nicely
unique_run_ids = filtered_df['run_name'].unique()

# Print them out line by line
for run_id in unique_run_ids:
    print(run_id)

# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    min_test_bet_rate ={'dog': 100/60, 'fav': 100/30},
)

# every two weeks is probably too infrequent for fave but also there might be noise in teh specific test group selected so don't want to lose a quality model. Might need to increase.
filtered_df[(filtered_df['bet_strat']=='fav')]

# In[ ]:




# In[ ]:


results_df = run_pipeline(
    wandb_project='nfl_bet_sweep_2',
    features=features,
    top_metric='loss',
    top_n=None,
    bet_strats=['dog','fav'],
    margins=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1],
    cy_df=cy_df
    , metric_threshold=.60
    # , exclude_tested = True
    , exclude_tested = False
    , pull_high_roi = True
)

# In[ ]:


filtered_df = filter_results_df(results_df, train_roi_min=7.5, val_test_roi_min=7.5)

# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    min_test_bet_rate ={'dog': 100/60, 'fav': 100/30},
)

# every two weeks is probably too infrequent for fave but also there might be noise in teh specific test group selected so don't want to lose a quality model. Might need to increase.
filtered_df[(filtered_df['bet_strat']=='fav')]

# In[ ]:


filtered_df[(filtered_df['bet_strat']=='dog')]

# In[ ]:


filtered_df = filter_results_df(results_df, train_roi_min=7.5, val_test_roi_min=7.5)

# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    min_test_bet_rate ={'dog': 100/30, 'fav': 100/15},
)

filtered_df[(filtered_df['bet_strat']=='fav')]

# In[ ]:


filtered_df[(filtered_df['bet_strat']=='dog')]

# In[ ]:


filtered_df = filter_results_df(results_df, train_roi_min=10, val_test_roi_min=10)

# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    min_test_bet_rate ={'dog': 100/60, 'fav': 100/30},
)

filtered_df[(filtered_df['bet_strat']=='fav')]

# In[ ]:


filtered_df[(filtered_df['bet_strat']=='dog')]

# In[ ]:


filtered_df = filter_results_df(results_df, train_roi_min=10, val_test_roi_min=10)

# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    min_test_bet_rate ={'dog': 100/60, 'fav': 100/15},
)

filtered_df[(filtered_df['bet_strat']=='fav')]

# In[ ]:


filtered_df[(filtered_df['bet_strat']=='dog')]

# In[ ]:




# In[ ]:


from functools import reduce
import pandas as pd

# --- CONFIG ---
bet_strat = 'fav'         # or 'dog'
base_bet = 100            # dollars per bet
# runs = {
#     # dauntless-sweep-42
#    '6po6v9zy': 0.055,
#     # jumping-sweep-111
#    'askoip2w': 0.065,
#     # genial-sweep-346
#    'biqlz8ar': 0.07,
#     # leafy-sweep-389
#    'j6q6o050': 0.065,
#     # magic-sweep-125
#    'm3jh0vdm': 0.065,
# }
runs = {
    # dauntless-sweep-42
   '6po6v9zy': 0.04,
    # jumping-sweep-111
   'askoip2w': 0.065,
    # genial-sweep-346
   'biqlz8ar': 0.06,
    # atomic-sweep-547
    'fs3i8481': 0.015,
    # leafy-sweep-389
   'j6q6o050': 0.065,
    # likely-sweep-678
    'pno9rh0z': 0.025
}

# keys we'll merge on
KEYS = ['season','week','fav_team','dog_team','dog_win']

# 1) First pass: grab the ground-truth profit column from any one run
first_run, first_margin = next(iter(runs.items()))
_, first_res = evaluate_single_run(
    wandb_project='nfl_bet_sweep_2',
    run_id=first_run,
    features=features,
    bet_strat=bet_strat,
    margin=first_margin
)
profit_col = f'{bet_strat}_profit'
profit_df = first_res['df'][KEYS + [profit_col]].rename(columns={profit_col: 'profit'})

# 2) Build one dataframe per run containing only KEYS + flag_{run_id}
flag_dfs = []
for run_id, margin in runs.items():
    _, res = evaluate_single_run(
        wandb_project='nfl_bet_sweep_2',
        run_id=run_id,
        features=features,
        bet_strat=bet_strat,
        margin=margin
    )
    df = res['df'][KEYS + ['bet']].copy()
    flag_col = f'flag_{run_id}'
    df[flag_col] = df['bet'] > 0
    flag_dfs.append(df[KEYS + [flag_col]])

# 3) Merge all flag dataframes on the identifying KEYS
merged_flags = reduce(
    lambda L, R: pd.merge(L, R, on=KEYS, how='outer'),
    flag_dfs
).fillna(False)   # missing flags → False

# 4) Attach the ground-truth profit column
merged = pd.merge(merged_flags, profit_df, on=KEYS, how='left').fillna({'profit': 0})

# 5) Compute per-run profit, bets, ROI
profit_per_run = {}
bets_per_run   = {}
roi_per_run    = {}

for run_id in runs:
    fcol = f'flag_{run_id}'
    p = merged.loc[merged[fcol], 'profit'].sum()
    b = merged[fcol].sum()
    profit_per_run[run_id] = p
    bets_per_run[run_id]   = int(b)
    roi_per_run[run_id]    = (p / (b * base_bet)) if b else float('nan')

print("Per-run results:")
for run_id in runs:
    print(f" • {run_id}: profit={profit_per_run[run_id]:.2f}, "
          f"bets={bets_per_run[run_id]}, "
          f"ROI={roi_per_run[run_id]:.4f}")

# 6) Compute ensemble metrics for "at least k" votes
merged['vote_count'] = merged[[f'flag_{r}' for r in runs]].sum(axis=1).astype(int)

ensemble = {}
N = len(runs)
for k in range(1, N+1):
    sub = merged[merged['vote_count'] >= k]   # <= change here
    profit_k = sub['profit'].sum()
    bets_k   = sub.shape[0]
    roi_k    = (profit_k / (bets_k * base_bet)) if bets_k else float('nan')
    ensemble[k] = {
        'profit': float(profit_k),
        'bets':   int(bets_k),
        'ROI':    float(roi_k)
    }

print("\nEnsemble (≥ k votes):")
for k, stats in ensemble.items():
    print(f" • k≥{k}: profit={stats['profit']:.2f}, "
          f"bets={stats['bets']}, ROI={stats['ROI']:.4f}")

# In[ ]:


merged[merged['vote_count']>=1]

# In[ ]:


merged[merged['vote_count']>=3]

# project 8

# In[ ]:


results_df = run_pipeline(
    # wandb_project='nfl_bet_sweep_2',
    wandb_project='nfl_bet_sweep_8',
    features=features,
    top_metric='loss',
    top_n=None,
    bet_strats=['dog','fav'],
    margins=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1],
    cy_df=cy_df
    , exclude_tested = False
    , pull_high_roi = True
)

# In[ ]:


filtered_df = filter_results_df(results_df, train_roi_min=7.5, val_test_roi_min=7.5)

# Show unique run_id values nicely
unique_run_ids = filtered_df['run_name'].unique()

# Print them out line by line
for run_id in unique_run_ids:
    print(run_id)

print(len(filtered_df))

filtered_df

# In[ ]:


# filtered_df = select_best_by_avg_roi(filtered_df)
filtered_df = select_best_by_avg_roi(
    filtered_df,
    # min_train_bet_rate={'dog': 0.02, 'fav': 0.05},
    min_test_bet_rate ={'dog': 100/60, 'fav': 100/30},
)
filtered_df

# In[ ]:


# every two weeks is probably too infrequent for fave but also there might be noise in teh specific test group selected so don't want to lose a quality model. Might need to increase.
filtered_df[(filtered_df['bet_strat']=='fav') & (filtered_df['test_bet_rate']>100/30)]
