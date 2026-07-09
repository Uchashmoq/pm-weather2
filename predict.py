import pandas as pd
import numpy as np
import datetime as dt
import config
import forecast
import real_time_data


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
    return -1


def bias_correction(fc, actual_temps: pd.DataFrame):
    df: pd.DataFrame = fc["data"].copy()
    fc_time = df["time"]
    start = get_temp_index(fc_time, actual_temps)
    if start == -1:
        return fc.copy()

    actual_temps = actual_temps.iloc[start:]
    for col_name in df.columns:
        if col_name == "time":
            continue

        fc_temps = df[col_name]
        df[col_name], _ = update_forecast(fc_time, fc_temps, actual_temps)

    updated_fc = fc.copy()
    updated_fc["data"] = df
    return updated_fc


def find_every_day_highest_temp(fc_time, fc_temps, timezone):
    """
    返回4列df，col0:fc_temps里每日最高温索引，col1每日最高温，col2每日最高温时间，col3日期
    fc_time代表每个温度的时间戳
    fc_temps代表未来数十小时的气温预测，每日最高温相当于局部最大值 [83.4, 85.1, 87.7, 90.6, 93.4, 95.8, 97.2, 97.6, 97.5, 96.7, 95.8, 94.6, 93.5, 92.2, 90.5, 88.7, 87, 85.6, 84.5, 83.9, 83.3, 83.2, 83.2, 83.7, 84.6, 86.3]
    """
    temps = np.asarray(fc_temps)
    times = np.asarray(fc_time)
    n = temps.size

    if n == 0:
        return pd.DataFrame(columns=["index", "temperature", "time", "date"])

    if isinstance(timezone, dict):
        timezone = timezone.get("timezone", "UTC+8")
    offset_seconds = int(str(timezone).upper().replace("UTC", "") or 0) * 3600

    local_days = np.floor_divide(times.astype(np.int64) + offset_seconds, 86400)
    _, day_starts = np.unique(local_days, return_index=True)
    day_ends = np.r_[day_starts[1:], n]
    highest_idx = np.empty(day_starts.size, dtype=np.intp)

    for i, (start, end) in enumerate(zip(day_starts, day_ends)):
        highest_idx[i] = start + np.argmax(temps[start:end])

    dates = [
        f"{d.strftime('%B').lower()}-{d.day}-{d.year}"
        for d in (
            dt.datetime.fromtimestamp(int(day) * 86400, tz=dt.timezone.utc)
            for day in local_days[highest_idx]
        )
    ]

    return pd.DataFrame(
        {
            "index": highest_idx,
            "temperature": temps[highest_idx],
            "time": times[highest_idx],
            "date": dates,
        }
    )


def fit_question(q, v, unit="C"):
    q1 = q
    if unit == "C":
        q1 = [t * 9 / 5 + 32 for t in q]
    if v < q1[0]:
        return q[0], 0
    if v >= q1[-1]:
        return q[-1], len(q) - 1
    for i in range(len(q) - 1):
        if q1[i] <= v < q1[i + 1]:
            return q[i], i
    return 0, 0


"""
report: date, outcome1, outcome2 ... 
"""


def predict_city(city, fc, actual_temps=None, questions=None, correction=True):
    if questions == None:
        questions = city["questions"]
    questions = list(questions)
    if actual_temps is not None and correction:
        fc = bias_correction(fc, actual_temps)
    df: pd.DataFrame = fc["data"]
    fc_time = df["time"].tolist()
    highest_temp_of_member = {}
    n_days = 9999
    for member in df.columns:
        if member == "time":
            continue

        fc_temps = df[member]
        # if member == "temperature_2m_member46":  # DEBUG
        #     print(fc_temps.tolist())
        ht = find_every_day_highest_temp(fc_time, fc_temps, city["timezone"])
        highest_temp_of_member[member] = ht
        n_days = min(n_days, ht.shape[0])

    report = pd.DataFrame(columns=["date"] + questions)
    for i in range(n_days):
        ocnt = [0] * len(questions)
        date = ""
        for member, ht in highest_temp_of_member.items():
            if date == "":
                date = ht["date"].iloc[i]
            temp = ht["temperature"].iloc[i]
            # if i == 0 and temp >= 94.3:  # DEBUG
            #     print(member)
            #     print(ht)
            #     print(temp)
            outcome, j = fit_question(questions, temp, city["temp_unit"])
            ocnt[j] += 1

        n_members = len(highest_temp_of_member)
        probs = [c / n_members for c in ocnt]
        report.loc[len(report)] = [date] + probs

    return report
