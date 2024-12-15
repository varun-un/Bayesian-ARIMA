import numpy as np
import pandas as pd

def invert_differencing(forecast_diff, d, original_series):
    """
    Reconstruct raw forecasted values from differenced forecasted values.

    Parameters:
        forecast_diff (array-like): Forecasted differenced values (Δ^d Y).
        d (int): Order of differencing.
        original_series (array-like): Original time series data (must have at least last d observations).

    Returns:
        raw_forecast (numpy.ndarray): Forecasted raw values on the original scale.
    """
    forecast_diff = np.asarray(forecast_diff)
    # original_series = pd.Series(original_series, index=np.arange(len(original_series)))

    if d < 0:
        raise ValueError("Order of differencing 'd' must be non-negative.")
    
    if d == 0:
        # return forecasts as-is
        return forecast_diff

    if len(original_series) < d:
        raise ValueError(f"Original series must have at least {d} observations for differencing order {d}.")

    # differencing d times to obtain the last d differenced values
    differenced_series = original_series.copy()
    for i in range(d):
        differenced_series = differenced_series.diff().dropna()

    # extract the last d differenced values required for inversion
    last_d_diffs = list(differenced_series.iloc[-d:].values)
    # reverse order to have highest order differenced value first
    last_d_diffs = last_d_diffs[::-1]

    raw_forecast = []

    for fd in forecast_diff:
        if d == 1:
            # first-order differencing -> add the forecast difference to the last value
            y_prev = original_series.iloc[-1] if len(raw_forecast) == 0 else raw_forecast[-1]
            y_new = y_prev + fd
            raw_forecast.append(y_new)
        else:
            # For higher-order differencing, propagate the differenced values appropriately
            # update first the highest order differenced value
            last_d_diffs[0] += fd

            # propagate the changes down to lower order differenced levels
            for i in range(1, d):
                last_d_diffs[i] += last_d_diffs[i-1]

            # end of the list gonna be the non-differenced value
            delta_y = last_d_diffs[-1]

            # compute the new raw forecasted value
            y_prev = original_series.iloc[-1] if len(raw_forecast) == 0 else raw_forecast[-1]
            y_new = y_prev + delta_y
            raw_forecast.append(y_new)

    return np.array(raw_forecast).flatten()     # flatten: (steps, 1) -> (steps,)
