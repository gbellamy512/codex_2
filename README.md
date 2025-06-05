# NFL Betting Project

This project converts an exploratory Jupyter notebook into a small Python package.
The package lives in `nfl_bet/` and exposes utilities for preparing data, training
models and evaluating betting strategies.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

Execute the main script to run a simple end-to-end pipeline:

```bash
python main.py
```

The script loads data from the `data/` directory, trains a logistic regression
model and prints the return on investment for a basic betting strategy.

## W&B Sweeps

Hyperparameter sweeps can be launched directly from the package:

```bash
python -m nfl_bet.wandb_train sweep
```

The sweep configuration mirrors the notebook setup. The following options can
be customised via command line flags or environment variables:

- `--project` / `WANDB_SWEEP_PROJECT` – W&B project name (default
  `nfl_bet_sweep`)
- `--count` / `WANDB_SWEEP_COUNT` – number of runs to execute (default `1`)

Example:

```bash
WANDB_SWEEP_PROJECT=my_proj python -m nfl_bet.wandb_train sweep --count 10
```

