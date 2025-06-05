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

