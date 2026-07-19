CITY = [
    {
        "name": "Chongqing-ZUCK",
        "lat": 29.718,
        "lon": 106.639,
        "timezone": "UTC+8",
        "ICAO": "ZUCK",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_seconds": 6 * 3600},
        ],
        "temp_unit": "C",
        "questions": range(30, 44),  # 30: 30 or below, 40: 40 or above, 37: 37<=T<38
    },
    {
        "name": "Chongqing-center",
        "lat": 29.57,
        "lon": 106.55,
        "timezone": "UTC+8",
        "ICAO": "ZUCK",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_seconds": 6 * 3600},
        ],
        "temp_unit": "C",
        "questions": range(30, 44),  # 30: 30 or below, 40: 40 or above, 37: 37<=T<38
    },
]

FETCH_AVIATIONWEATHER_INTERVAL = 600
FETCH_FORECAST_INTERVAL = 600
UPDATE_PREDICTION_INTERVAL = 800
