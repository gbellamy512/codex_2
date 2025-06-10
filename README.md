# NFL Betting Model

## Overview

This project is designed to predict NFL game outcomes and to evaluate associated betting strategies. It leverages a comprehensive set of historical NFL data, including Expected Points Added (EPA) metrics, team pass rates, weekly win percentages, and detailed game schedules. The system employs machine learning models, with training and hyperparameter optimization managed by Weights & Biases (wandb), to generate predictive outputs. These outputs are then utilized within a simulated betting environment to rigorously assess the profitability of different betting strategies against historical betting lines.

---
## Key Features

* **Comprehensive Data Integration**:
    * Combines diverse NFL datasets, including historical and current year EPA data.
    * Incorporates team-specific statistics like pass rates and evolving win percentages.
    * Utilizes detailed game schedules, incorporating information on betting lines, team rest days, and factors like divisional matchups.

* **Advanced Feature Engineering**:
    * Calculates dynamic team statistics using methods such as Exponentially Weighted Moving Averages (EWMA) or simple season averages.
    * Determines head-to-head (H2H) matchup history within a given season to capture recent performance nuances.
    * Analyzes betting lines to derive metrics like implied probabilities.
    * Constructs "advantage" columns by comparing relative team strengths across various offensive, defensive, and special teams categories.

* **Machine Learning Model Training**:
    * Employs TensorFlow/Keras for building neural network models capable of predicting specified game-related outcomes.
    * Integrates Weights & Biases (`wandb`) for systematic hyperparameter sweeping, facilitating optimization of model architecture (e.g., layers, neurons), training parameters (e.g., learning rate, batch size), and regularization techniques (e.g., dropout, L1/L2).

* **Betting Strategy Simulation**:
    * Evaluates model predictions against historical betting lines (e.g., moneylines, spreads).
    * Calculates key performance indicators such as potential profit and Return on Investment (ROI) for simulated bets.
    * Supports flexible betting decision logic, including strategies based on expected value (comparing model-derived probabilities to sportsbook-implied probabilities plus a margin) or threshold-based betting from model outputs.
    * Allows for targeted betting approaches based on various criteria derived from the model's predictions.

* **Robust Evaluation Pipeline**:
    * Retrieves top-performing model configurations from `wandb` experiments based on user-defined metrics.
    * Efficiently loads versioned model artifacts, including trained models and their corresponding preprocessing pipelines, from `wandb`.
    * Ensures consistent data splitting (training, validation, test, and current-year sets) by utilizing run-specific random states logged during training.
    * Systematically evaluates and compares the performance of numerous betting strategies and associated parameters (e.g., margins) across different data splits.
    * Facilitates experiment tracking by tagging `wandb` runs based on their evaluation outcomes (e.g., 'tested', 'high_roi_strategy_X').

* **Artifact Management**:
    * Leverages `wandb` for logging, versioning, and retrieving critical artifacts such as preprocessing pipelines and trained machine learning models, ensuring reproducibility and traceability.

## W&B Usage

Evaluation and training utilities use Weights & Biases for experiment tracking. If your runs live under a specific W&B entity (organization or username), set the `WANDB_ENTITY` environment variable so the scripts can locate artifacts and runs across accounts.

## Running the Scripts

### `main.py`

Quickly train a model and evaluate ROI. See
[Orientation and Bet Type](#orientation-and-bet-type) for how the
`--orientation` and `--bet-type` options affect the targets.

| Argument | Default | Description |
|----------|---------|-------------|
| `--orientation` | `fav_dog` | matchup framing |
| `--bet-type` | `moneyline` | wager type |
| `--save-csv` | `False` | write `results/betting_results_<orientation>_<bet-type>.csv` |

Minimal example:

```bash
python main.py
```

All parameters:

```bash
python main.py --orientation home_away --bet-type spread --save-csv
```
This writes `results/betting_results_home_away_spread.csv`.

### `python -m nfl_bet.wandb_train`

Utilities for running training jobs with optional W&B sweeps.

**example** subcommand

| Argument | Default |
|----------|---------|
| `--orientation` | `fav_dog` |
| `--bet-type` | `moneyline` |

**sweep** subcommand

| Argument | Default |
|----------|---------|
| `--project` | env `WANDB_SWEEP_PROJECT` or `nfl_bet_sweep` |
| `--count` | env `WANDB_SWEEP_COUNT` or `1` |
| `--orientation` | `fav_dog` |
| `--bet-type` | `moneyline` |

Minimal example:

```bash
python -m nfl_bet.wandb_train example
```

All parameters:

```bash
python -m nfl_bet.wandb_train sweep --project nfl_bet_sweep --count 3 --orientation home_away --bet-type spread
```

### `python -m nfl_bet.wandb_eval`

Evaluate the top runs of a W&B project.

| Argument | Default |
|----------|---------|
| `--project` | `nfl_bet_sweep_8` |
| `--top-metric` | `loss` |
| `--top-n` | `10` |
| `--train-weight` | `1.0` |
| `--metric-threshold` | `0.60` |
| `--exclude-tested` | `False` |
| `--pull-high-roi` | `False` |
| `--orientation` | `fav_dog` |
| `--bet-type` | `moneyline` |

Minimal example:

```bash
python -m nfl_bet.wandb_eval run
```

All parameters:

```bash
python -m nfl_bet.wandb_eval run --project nfl_bet_sweep_8 --top-metric loss --top-n 10 --train-weight 1.0 --metric-threshold 0.60 --exclude-tested --pull-high-roi --orientation home_away --bet-type spread
```
### Orientation and Bet Type

The training and evaluation commands accept two important arguments:

* `--orientation` selects which side of each matchup is modeled. Use `fav_dog` to frame games by favorite and underdog or `home_away` to model home and away teams directly.
* `--bet-type` chooses whether the targets are based on straight wins (`moneyline`) or covering the spread (`spread`).

Depending on these choices, different target columns and odds are used. Implied probabilities are derived with `calculate_implied_probabilities()` from the respective odds columns.

| orientation | bet_type | target | odds columns |
|-------------|---------|--------|--------------|
| `fav_dog`   | `moneyline` | `dog_win`   | `dog_moneyline` / `fav_moneyline` |
| `fav_dog`   | `spread`    | `fav_cover` | `fav_spread_odds` / `dog_spread_odds` |
| `home_away` | `moneyline` | `home_win`  | `home_moneyline` / `away_moneyline` |
| `home_away` | `spread`    | `home_cover`| `home_spread_odds` / `away_spread_odds` |

For spread bets, ``get_betting_context()`` also returns ``line_col`` and
``regression_target``. These point to the appropriate spread column (e.g.,
``fav_line`` or ``home_line``) and the numeric margin target (``fav_margin`` or
``home_margin``).

When ``--bet-type spread`` is selected the CLI tools automatically switch to a
regression model. The network's last layer uses a linear activation and MSE loss
while predictions represent the expected margin. Betting decisions compare this
margin against the market line plus any supplied threshold.

#### Training a home/away spread model

```bash
python -m nfl_bet.wandb_train example --orientation home_away --bet-type spread
```

#### Evaluating ROI for that model

```bash
python -m nfl_bet.wandb_eval run \
    --project nfl_bet_sweep_8 \
    --orientation home_away \
    --bet-type spread
```
