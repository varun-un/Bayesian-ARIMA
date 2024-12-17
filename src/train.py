import argparse
import sys
from pathlib import Path
import pandas as pd

from model import HierarchicalModel
from ensemble import WeightedAverageEnsemble

def main():
    parser = argparse.ArgumentParser(description='Train a HierarchicalModel for a given ticker.')
    parser.add_argument('ticker', type=str, help='Ticker symbol of the stock (e.g. AAPL, MSFT)')
    parser.add_argument('--num_draws', type=int, default=300, help='Number of draws for MCMC sampling')
    parser.add_argument('--num_tune', type=int, default=100, help='Number of tuning steps for MCMC sampling')
    parser.add_argument('--target_accept', type=float, default=0.95, help='Target acceptance rate for MCMC')

    args = parser.parse_args()

    ticker = args.ticker
    num_draws = args.num_draws
    num_tune = args.num_tune
    target_accept = args.target_accept

    print(f"Training HierarchicalModel for ticker: {ticker} with num_draws={num_draws}, num_tune={num_tune}, target_accept={target_accept}")

    WA = WeightedAverageEnsemble([0.3, 0.4, 0.3])

    model = HierarchicalModel(ticker=ticker, ensemble=WA, memory_save=True)
    model.train_models(num_draws=num_draws, num_tune=num_tune, target_accept=target_accept)
    model.save()

    print("Model training completed and saved successfully.")

if __name__ == '__main__':
    main()

    # python3 train.py AAPL --num_draws 500 --num_tune 200 --target_accept 0.9
    # python3 src/train.py MSFT --num_draws 200 --num_tune 200 --target_accept 0.9
