import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .model import HierarchicalModel
from src.utils import fetch_all_data, TradingTimeDelta

def main():
    parser = argparse.ArgumentParser(description='Predict future values using a HierarchicalModel.')
    parser.add_argument('ticker', type=str, help='Ticker symbol of the stock')
    parser.add_argument('end_time', type=str, help='End time for the prediction (e.g. "2025-01-15 10:30:00")')
    args = parser.parse_args()

    ticker = args.ticker
    end_time_str = args.end_time
    end_time = pd.to_datetime(end_time_str)

    # load the hierarchical model
    model = HierarchicalModel(ticker=ticker)
    model.load()  # uses default filename convention

    delta_t = TradingTimeDelta(start_time=pd.Timestamp.now(), end_time=end_time)

    # predict the single future value via ensemble
    ensemble_value = model.predict_value(end_time)

    # get the predicted series (labelled_forecasts) for visualization
    predictions, labelled_forecasts = model.predict_to_time(end_time)

    interval = '1h' if delta_t.delta_t > 24 * 60 * 60 else '1m'
    label = 'hourly' if interval == '1h' else 'minute'

    # get delta_t of historical data
    historical_data = fetch_all_data(ticker, delta_t.get_delta_minutes * 60, model.fetch_range[label], interval, end_date=int(pd.Timestamp.now().timestamp()))
    historical_data = historical_data.dropna()

    # Combine historical and future data for plotting
    plt.figure(figsize=(12, 6))

    # Plot historical close prices
    plt.plot(historical_data.index, historical_data['Close'], label='Historical', color='black', linewidth=2)

    # plot each model's forecast
    colors = {'daily': 'blue', 'hourly': 'green', 'minute': 'red'}
    for timeframe, forecast_series in labelled_forecasts.items():
        # forecast_series is indexed by future times
        # some future indexing might have shifted due to interpolation
        forecast_series = forecast_series.dropna()
        if len(forecast_series) > 0:
            plt.plot(forecast_series.index, forecast_series.values, label=f'{timeframe} forecast', color=colors.get(timeframe, 'gray'), linestyle='--')

    # add the ensemble prediction as a single point at end_time
    plt.plot(end_time, ensemble_value, 'o', color='magenta', label='Ensemble Prediction', markersize=10)

    plt.title(f"Predictions for {ticker} at {end_time_str}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print the ensemble prediction for reference
    print(f"Ensemble Prediction at {end_time_str}: {ensemble_value}")

if __name__ == '__main__':
    main()
