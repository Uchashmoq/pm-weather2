from data_source import aviationweather_temperature
import config
import atexit
import asyncio
import logging
import pandas as pd
import pickle
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

temperature_history = {}
BASE_DIR = Path(__file__).resolve().parent
TEMPERATURE_DIR = BASE_DIR / "data" / "temperature"
TEMPERATURE_HISTORY_FILE = TEMPERATURE_DIR / "temperature_history.pkl"


def update_temperature():
    new_temperatures = []
    temperatures_by_station = {}
    update_time = int(time.time())
    for city in config.CITY:
        city_name = city["name"]
        station = city["ICAO"].strip().upper()
        if station not in temperatures_by_station:
            temperatures_by_station[station] = aviationweather_temperature(station)
        temp = temperatures_by_station[station]

        if city_name not in temperature_history:
            temperature_history[city_name] = pd.DataFrame(
                columns=["update_time", "temperature"]
            )

        temperature_history[city_name].loc[len(temperature_history[city_name])] = [
            update_time,
            temp,
        ]
        new_temperatures.append(
            {"city": city_name, "update_time": update_time, "temperature": temp}
        )
    return bool(new_temperatures), new_temperatures


def save_temperature():
    TEMPERATURE_DIR.mkdir(parents=True, exist_ok=True)
    with TEMPERATURE_HISTORY_FILE.open("wb") as f:
        pickle.dump(temperature_history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_temperature():
    if not TEMPERATURE_HISTORY_FILE.exists():
        return temperature_history

    with TEMPERATURE_HISTORY_FILE.open("rb") as f:
        saved_history = pickle.load(f)

    temperature_history.clear()
    temperature_history.update(saved_history)
    return temperature_history


def _save_temperature_on_signal(signum, frame):
    save_temperature()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    sys.exit(128 + signum)


atexit.register(save_temperature)
signal.signal(signal.SIGINT, _save_temperature_on_signal)
signal.signal(signal.SIGTERM, _save_temperature_on_signal)


async def update_temperature_periotically():
    load_temperature()
    if not temperature_history and logger.isEnabledFor(logging.INFO):
        logger.info("temperature history is empty")
    while True:
        try:
            updated, new_temperatures = update_temperature()
            if updated:
                save_temperature()
            if new_temperatures and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "new temperatures: count=%d, temperatures=%s",
                    len(new_temperatures),
                    new_temperatures,
                )

        except Exception:
            logger.exception(f"failed to update temperature")
        await asyncio.sleep(config.FETCH_AVIATIONWEATHER_INTERVAL)
