from data_source import ensemble_forcast, wunderground_temperature
import config
import pandas as pd
import time

temperature_history = {}


def update_temperature():
    for city in config.CITY:
        city_name = city["name"]
        wurl = city["wunder_url"]
        temp = wunderground_temperature(wurl)
        update_time = int(time.time())

        if city_name not in temperature_history:
            temperature_history[city_name] = pd.DataFrame(
                columns=["update_time", "temperature"]
            )

        temperature_history[city_name].loc[len(temperature_history[city_name])] = [
            update_time,
            temp,
        ]
