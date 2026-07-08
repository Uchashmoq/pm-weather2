import atexit
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import forecast
import predict
import real_time_data

try:
    atexit.unregister(forecast.save_forecast)
except ValueError:
    pass


MOCK_FILE = Path(__file__).with_name("essemble_api_mock.json")


def mock_forecast(*_args, **_kwargs):
    payload = json.loads(MOCK_FILE.read_text())
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("int64") // 1_000_000_000
    return {
        "timestamp": int(df["time"].iloc[0]),
        "lat": payload["latitude"],
        "lon": payload["longitude"],
        "data": df,
    }


class PredictTest(unittest.TestCase):
    def test_update_forecast_applies_decaying_bias_to_future_points(self):
        fc_time = pd.Series([0, 3600, 7200, 10800], dtype="float64")
        fc_temps = pd.Series([10, 11, 12, 13], dtype="float64")
        actual_temps = pd.DataFrame(
            {
                "update_time": [3600, 7200],
                "temperature": [13, 14],
            }
        )

        updated, bias = predict.update_forecast(fc_time, fc_temps, actual_temps)

        self.assertAlmostEqual(bias, 2.0)
        self.assertEqual(updated.iloc[0], 10)
        self.assertEqual(updated.iloc[1], 11)
        self.assertAlmostEqual(updated.iloc[2], 14.0)
        self.assertGreater(updated.iloc[3], 13)
        self.assertLess(updated.iloc[3], 15)

    def test_bias_correction_returns_new_forecast_with_updated_dataframe(self):
        fc = mock_forecast()
        temp_col = "temperature_2m"
        first_time = int(fc["data"]["time"].iloc[0])
        actual_temps = pd.DataFrame(
            {
                "update_time": [first_time],
                "temperature": [fc["data"][temp_col].iloc[0] + 3],
            }
        )

        updated = predict.bias_correction(fc, actual_temps)

        self.assertIsNot(updated, fc)
        self.assertIsNot(updated["data"], fc["data"])
        self.assertEqual(updated["timestamp"], fc["timestamp"])
        self.assertEqual(fc["data"][temp_col].iloc[0], 82.1)
        self.assertAlmostEqual(updated["data"][temp_col].iloc[0], 85.1)


class ForecastUpdateTest(unittest.TestCase):
    def setUp(self):
        forecast.forecast_history.clear()
        self.addCleanup(forecast.forecast_history.clear)

    def test_update_forecast_uses_ensemble_data_and_records_new_forecast(self):
        city_config = [
            {
                "name": "TestCity",
                "lat": 29.75,
                "lon": 106.75,
                "models": [{"name": "mock_model"}],
            }
        ]

        with (
            patch.object(forecast.config, "CITY", city_config),
            patch.object(forecast, "ensemble_forcast", side_effect=mock_forecast) as ensemble,
            patch.object(forecast.time, "time", return_value=123456),
        ):
            updated, new_forecasts = forecast.update_forecast()
            updated_again, new_forecasts_again = forecast.update_forecast()

        self.assertTrue(updated)
        self.assertEqual(len(new_forecasts), 1)
        self.assertEqual(new_forecasts[0]["city"], "TestCity")
        self.assertEqual(new_forecasts[0]["update_time"], 123456)
        self.assertIn("TestCity", forecast.forecast_history)
        self.assertIn("mock_model", forecast.forecast_history["TestCity"])
        self.assertFalse(updated_again)
        self.assertEqual(new_forecasts_again, [])
        ensemble.assert_called_with(29.75, 106.75, "mock_model")


class RealTimeDataTest(unittest.TestCase):
    def setUp(self):
        real_time_data.temperature_history.clear()
        self.addCleanup(real_time_data.temperature_history.clear)

    def test_update_temperature_appends_wunderground_temperature(self):
        city_config = [
            {
                "name": "TestCity",
                "wunder_url": "https://example.test/weather",
            }
        ]

        with (
            patch.object(real_time_data.config, "CITY", city_config),
            patch.object(real_time_data, "wunderground_temperature", return_value=88.5),
            patch.object(real_time_data.time, "time", return_value=654321),
        ):
            real_time_data.update_temperature()

        temps = real_time_data.temperature_history["TestCity"]
        self.assertEqual(list(temps.columns), ["update_time", "temperature"])
        self.assertEqual(temps.iloc[0].to_dict(), {"update_time": 654321.0, "temperature": 88.5})


if __name__ == "__main__":
    unittest.main()
