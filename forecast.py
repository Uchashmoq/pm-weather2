from data_source import ensemble_forcast
import config
import atexit
import asyncio
import logging
import pickle
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

forecast_history = {}
BASE_DIR = Path(__file__).resolve().parent
FORECAST_DIR = BASE_DIR / "data" / "forecast"
FORECAST_HISTORY_FILE = FORECAST_DIR / "forecast_history.pkl"


def forecast_eq(fc1, fc2):
    return fc1["data"].equals(fc2["data"])


def update_forecast():
    updated = False
    new_fc = []
    for city in config.CITY:
        lat = city["lat"]
        lon = city["lon"]
        city_name = city["name"]
        for model in city["models"]:
            model_name = model["name"]
            fc = ensemble_forcast(lat, lon, model_name)
            fc["city"] = city_name
            fc["update_time"] = int(time.time())
            if city_name not in forecast_history:
                forecast_history[city_name] = {}

            if model_name not in forecast_history[city_name]:
                forecast_history[city_name][model_name] = []

            fcs: list = forecast_history[city_name][model_name]
            if len(fcs) == 0 or not forecast_eq(fc, fcs[-1]):
                fcs.append(fc)
                new_fc.append(fc)
                updated = True
    return updated, new_fc


def save_forecast():
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    with FORECAST_HISTORY_FILE.open("wb") as f:
        pickle.dump(forecast_history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_forecast():
    if not FORECAST_HISTORY_FILE.exists():
        return forecast_history

    with FORECAST_HISTORY_FILE.open("rb") as f:
        saved_history = pickle.load(f)

    forecast_history.clear()
    forecast_history.update(saved_history)
    return forecast_history


def _save_forecast_on_signal(signum, frame):
    save_forecast()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    sys.exit(128 + signum)


atexit.register(save_forecast)
signal.signal(signal.SIGINT, _save_forecast_on_signal)
signal.signal(signal.SIGTERM, _save_forecast_on_signal)


async def update_forecast_periotically():
    load_forecast()
    if not forecast_history and logger.isEnabledFor(logging.INFO):
        logger.info("forecast history is empty")
    while True:
        try:
            updated, new_forecasts = update_forecast()
            if updated:
                save_forecast()
            if new_forecasts and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "new forecasts: count=%d, forecasts=%s",
                    len(new_forecasts),
                    [
                        {
                            "city": fc["city"],
                            "timestamp": time.strftime(
                                "%H:%M", time.localtime(fc["timestamp"])
                            ),
                        }
                        for fc in new_forecasts
                    ],
                )
        except Exception:
            logger.exception("failed to update forecast")

        await asyncio.sleep(config.FETCH_FORECAST_INTERVAL)


if __name__ == "__main__":
    # update_forecast()
    # print(forecast_history)
    # update_temperature()
    # print(temperature_history)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(update_forecast_periotically())
