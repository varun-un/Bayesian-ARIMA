import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


from model import HierarchicalModel
from utils import fetch_all_data, TradingTimeDelta

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

    # get the predicted series (labelled_forecasts) for visualization
    predictions, labelled_forecasts = model.predict_to_time(end_time)

    # find any predictions that are extraneous values (model diverged)
    problems = None
    for timeframe, prediction in predictions.items():
        if not np.isfinite(prediction):
            print(f"WARNING: {timeframe} model prediction is not a finite number. Discarding")
            problems = timeframe
        if prediction < 0:
            print(f"WARNING: {timeframe} model prediction is negative. Discarding")
            problems = timeframe
        if prediction > 1e6:
            print(f"WARNING: {timeframe} model prediction is very large. Discarding")
            problems = timeframe

    # remove any problematic predictions
    if problems:
        del labelled_forecasts[problems]
        del predictions[problems]

        # re-align weights to set the problematic model to 0
        if problems == 'daily':
            model.ensemble.set_weights([0, 0.4, 0.6])
        elif problems == 'hourly':
            model.ensemble.set_weights([0.5, 0, 0.5])
        elif problems == 'minute':
            model.ensemble.set_weights([0.4, 0.6, 0])

    # predict the single future value via ensemble
    ensemble_value = model.predict_value(end_time)

    interval = '1h' if delta_t.get_delta_minutes() > 24 * 60 else '1m'
    label = 'hourly' if interval == '1h' else 'minute'
    range = max(delta_t.get_delta_minutes() * 60, 4*24*60*60)       # make sure we have at least 4 days of data, in case of weekends

    # get delta_t of historical data 
    historical_data = fetch_all_data(ticker, 
                                     range, 
                                     int(model.fetch_range[label]) if model.fetch_range[label] is not None else model.fetch_range[label], 
                                     interval, 
                                     end_date=int(pd.Timestamp.now().timestamp())
                                     )
    historical_data = historical_data.dropna()

    plt.figure(figsize=(12, 6))

    # Plot historical close prices
    plt.plot(historical_data.index, historical_data['Close'], label='Historical', color='black', linewidth=1)

    # plot each model's forecast
    colors = {'daily': 'blue', 'hourly': 'green', 'minute': 'red'}
    for timeframe, forecast_series in labelled_forecasts.items():
        # forecast_series is indexed by future times
        print(f"Forecast for {timeframe}:")
        print(forecast_series)
        forecast_series = forecast_series.dropna()
        if len(forecast_series) > 0:
            plt.plot(forecast_series.index, forecast_series.values, label=f'{timeframe} forecast', color=colors.get(timeframe, 'gray'), linestyle='--')

    # add the ensemble prediction as a single point at the end_time
    plt.plot(end_time, ensemble_value, 'o', color='magenta', label='Ensemble Prediction', markersize=10)

    # x-axis to show only business days
    ax = plt.gca()  
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # Set ticks for Mondays
    ax.xaxis.set_minor_locator(mdates.DayLocator())  #  add minor ticks for every day
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Y-%m-%d'))  # format minor ticks as "YYYY-MM-DD" as well
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # format major ticks as "YYYY-MM-DD"

    plt.title(f"Predictions for {ticker} at {end_time_str}")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
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

    #  python3 src/predict.py AAPL "2024-12-18 10:30:00"
