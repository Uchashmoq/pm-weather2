import pandas as pd
import numpy as np


# fc_time: timestamp of every forecast,fc_temps: temperature in every forecast ,actual_temps: DataFrame with update_time and temperature columns
def update_forecast(fc_time, fc_temps, actual_temps: pd.DataFrame):
    fc_time_arr = fc_time.to_numpy(dtype=np.float64, copy=False)
    fc_temp_arr = fc_temps.to_numpy(dtype=np.float64, copy=False)
    actual_time = actual_temps["update_time"].to_numpy(dtype=np.float64, copy=False)
    actual_temp = actual_temps["temperature"].to_numpy(dtype=np.float64, copy=False)
    if len(fc_time_arr) < 2 or len(actual_time) == 0:
        return fc_temps, 0.0

    fc_steps = np.diff(fc_time_arr)
    fc_steps = fc_steps[fc_steps > 0]
    if len(fc_steps) == 0:
        return fc_temps, 0.0

    fc_step = float(np.median(fc_steps))
    interval_idx = np.searchsorted(fc_time_arr, actual_time, side="right") - 1
    valid = (interval_idx >= 0) & (interval_idx + 1 < len(fc_time_arr))
    if not np.any(valid):
        return fc_temps, 0.0

    interval_idx = interval_idx[valid]
    actual_time = actual_time[valid]
    actual_temp = actual_temp[valid]

    fct1 = fc_time_arr[interval_idx]
    fct2 = fc_time_arr[interval_idx + 1]
    fc_temp1 = fc_temp_arr[interval_idx]
    fc_temp2 = fc_temp_arr[interval_idx + 1]
    fc_temp = fc_temp1 + (fc_temp2 - fc_temp1) * (actual_time - fct1) / (fct2 - fct1)
    errs = actual_temp - fc_temp

    latest_actual_time = actual_time[-1]
    bias_tau = 3.0 * fc_step
    err_weights = np.exp(-(latest_actual_time - actual_time) / bias_tau)
    bias = float(np.average(errs, weights=err_weights))

    decay_tau = 6.0 * fc_step
    future_idx = np.flatnonzero(fc_time_arr >= latest_actual_time)
    if len(future_idx) == 0:
        return fc_temps, bias

    hours_decay = np.exp(-(fc_time_arr[future_idx] - latest_actual_time) / decay_tau)
    fc_temps.iloc[future_idx] = fc_temp_arr[future_idx] + bias * hours_decay
    return fc_temps, bias


def get_temp_index(fc_time, actual_temps):
    t1 = fc_time[0]
    for i in range(actual_temps.shape[0]):
        t = actual_temps["update_time"].iat[i]
        if t >= t1:
            return i
    raise RuntimeError(f"can't find the first real temperature")


def bias_correction(fc, actual_temps: pd.DataFrame):
    df: pd.DataFrame = fc["data"].copy()
    fc_time = df["time"]
    start = get_temp_index(fc_time, actual_temps)
    actual_temps = actual_temps.iloc[start:]
    for col_name in df.columns:
        if col_name == "time":
            continue

        fc_temps = df[col_name]
        df[col_name], _ = update_forecast(fc_time, fc_temps, actual_temps)

    updated_fc = fc.copy()
    updated_fc["data"] = df
    return updated_fc
