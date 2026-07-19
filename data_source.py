import math

import requests
import pandas as pd

AVIATIONWEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
REQUEST_TIMEOUT_SECONDS = 15
REQUEST_HEADERS = {
    "User-Agent": "pm-weather2/1.0",
    "Accept": "application/json",
}


def aviationweather_temperature(icao: str) -> float:
    """Return the latest METAR temperature for an ICAO station in Celsius."""
    station = icao.strip().upper()
    if not station:
        raise ValueError("ICAO station identifier must not be empty")

    response = requests.get(
        AVIATIONWEATHER_METAR_URL,
        params={"ids": station, "format": "json"},
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    if response.status_code == 204:
        raise RuntimeError(f"No METAR data available for {station}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid METAR response for {station}") from exc

    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"No METAR data available for {station}")

    observation = next(
        (
            item
            for item in payload
            if isinstance(item, dict)
            and str(item.get("icaoId", "")).upper() == station
        ),
        None,
    )
    if observation is None and len(payload) == 1 and isinstance(payload[0], dict):
        observation = payload[0]
    if observation is None:
        raise RuntimeError(f"No METAR data available for {station}")

    temperature = observation.get("temp")
    if temperature is None or isinstance(temperature, bool):
        raise RuntimeError(f"METAR temperature is unavailable for {station}")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid METAR temperature for {station}") from exc
    if not math.isfinite(temperature):
        raise RuntimeError(f"Invalid METAR temperature for {station}")
    return temperature


def ensemble_forcast(lat=29.75, lon=106.75, model="ecmwf_aifs025_ensemble"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": [model],
        "forecast_days": 3,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }
    resp = requests.get(
        url="https://ensemble-api.open-meteo.com/v1/ensemble", params=params
    ).json()

    df = pd.DataFrame()
    for k, v in resp["hourly"].items():
        # Temperature columns are Celsius; time contains Unix timestamps.
        new_cols = pd.DataFrame({k: v})
        df = pd.concat([df, new_cols], axis=1)
    return {
        "timestamp": int(df["time"].iloc[0]),
        "lat": resp["latitude"],
        "lon": resp["longitude"],
        "data": df,
    }


if __name__ == "__main__":
    df = ensemble_forcast()
    print(df)
