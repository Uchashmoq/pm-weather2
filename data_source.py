import re
import requests
import pandas as pd


def claw_wunderground(url, pattern):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    html = requests.get(url, headers=headers, timeout=15).text
    m = re.search(pattern, html)
    if m:
        temp = int(m.group(1))
        return temp
    else:
        raise RuntimeError("No data")


def wunderground_temperature(url: str) -> int:
    pattern = (
        r'<span\b(?=[^>]*class="[^"]*\bwu-unit-temperature\b[^"]*")[^>]*>'
        r"[\s\S]*?"
        r'<span\b(?=[^>]*class="[^"]*\bwu-value\b[^"]*\bwu-value-to\b[^"]*")[^>]*>'
        r"\s*(-?\d+)\s*</span>"
    )
    return claw_wunderground(url, pattern)


def wunderground_tomorrow_high(url: str) -> int:
    pattern = (
        r'<span\b[^>]*class="[^"]*\bday\b[^"]*"[^>]*>\s*Tomorrow\s*</span>'
        r"[\s\S]*?"
        r'<span\b[^>]*class="[^"]*\bwu-value\b[^"]*\bwu-value-to\b[^"]*"[^>]*>'
        r"\s*(-?\d+)\s*</span>\s*&nbsp;"
    )

    return claw_wunderground(url, pattern)


def ensemble_forcast(lat=29.75, lon=106.75, model="ecmwf_aifs025_ensemble"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": [model],
        "forecast_days": 5,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
        "temperature_unit": "fahrenheit",
    }
    resp = requests.get(
        url="https://ensemble-api.open-meteo.com/v1/ensemble", params=params
    ).json()
    df = pd.DataFrame()
    for k, v in resp["hourly"].items():
        # k:"temperature_2m_ecmwf_aifs025_ensemble", v: [81.7, 83.2, 85.4, 87.8, 90.3...]   or k: "time", v: [1783382400, 1783386000, 1783389600, 1783393200, 1783396800 ... ]
        new_cols = pd.DataFrame({k: v})
        df = pd.concat([df, new_cols], axis=1)
    return {
        "timestamp": int(df["time"].iloc[0]),
        "lat": resp["latitude"],
        "lon": resp["longitude"],
        "data": df,
    }


if __name__ == "__main__":
    # url = "https://www.wunderground.com/weather/cn/chongqing/ZUCK"
    # # url = "https://www.wunderground.com/weather/kr/incheon/RKSI"
    # t1 = wunderground_temperature(url)
    # t2 = wunderground_tomorrow_high(url)
    # print(f"{url}\nNow: {t1} F\nTomorrow: {t2}F")
    df = ensemble_forcast()
    print(df)
