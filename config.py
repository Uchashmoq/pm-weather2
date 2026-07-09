CITY = [
    {
        "name": "Chongqing",
        "lat": 29.57,
        "lon": 106.55,
        "timezone": "UTC+8",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_seconds": 6 * 3600},
        ],
        "wunder_url": "https://www.wunderground.com/weather/cn/chongqing/ZUCK",
        "temp_unit": "C",
        "questions": range(28, 43),  # 30: 30 or below, 40: 40 or above, 37: 37<=T<38
    }
]

FETCH_WUNDERGROUND_INTARVAL = 150
FETCH_FORECAST_INTERVAL = 120
UPDATE_PREDICTION_INTERVAL = 600
