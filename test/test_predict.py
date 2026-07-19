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
import data_source
import predict
import real_time_data

for save_callback in (forecast.save_forecast, real_time_data.save_temperature):
    try:
        atexit.unregister(save_callback)
    except ValueError:
        pass


MOCK_FILE = Path(__file__).with_name("essemble_api_mock.json")


def mock_forecast(*_args, **_kwargs):
    payload = json.loads(MOCK_FILE.read_text())
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("int64") // 1_000_000
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

        self.assertEqual(
            list(bias.columns), ["time", "actual_temp", "forecast_temp"]
        )
        self.assertTrue((bias["actual_temp"] - bias["forecast_temp"] == 2.0).all())
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

        updated, member_bias = predict.bias_correction(fc, actual_temps)

        self.assertIsNot(updated, fc)
        self.assertIsNot(updated["data"], fc["data"])
        self.assertEqual(updated["timestamp"], fc["timestamp"])
        self.assertEqual(fc["data"][temp_col].iloc[0], 27.1)
        self.assertAlmostEqual(updated["data"][temp_col].iloc[0], 30.1)
        self.assertIn(temp_col, member_bias)

    def test_fit_question_compares_celsius_values_directly(self):
        questions = list(range(28, 43))

        self.assertEqual(predict.fit_question(questions, 31.9, "C"), (31, 3))
        self.assertEqual(predict.fit_question(questions, 42.0, "C"), (42, 14))

    def test_find_every_day_highest_temp_prints_mock_result(self):
        fc = mock_forecast()
        city = {"timezone": "UTC+8"}
        result = predict.find_every_day_highest_temp(
            fc["data"]["time"],
            fc["data"]["temperature_2m"],
            city,
        )

        print("\nfind_every_day_highest_temp mock result:")
        print(result)

        self.assertEqual(list(result.columns), ["index", "temperature", "time", "date"])
        self.assertFalse(result.empty)
        self.assertEqual(result["date"].nunique(), len(result))


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

class DataSourceTest(unittest.TestCase):
    def test_aviationweather_temperature_returns_celsius_metar_value(self):
        with patch.object(data_source.requests, "get") as request:
            response = request.return_value
            response.status_code = 200
            response.json.return_value = [{"icaoId": "ZUCK", "temp": 31}]

            temperature = data_source.aviationweather_temperature(" zuck ")

        self.assertEqual(temperature, 31.0)
        response.raise_for_status.assert_called_once_with()
        request.assert_called_once_with(
            data_source.AVIATIONWEATHER_METAR_URL,
            params={"ids": "ZUCK", "format": "json"},
            headers=data_source.REQUEST_HEADERS,
            timeout=data_source.REQUEST_TIMEOUT_SECONDS,
        )

    def test_aviationweather_temperature_rejects_missing_data(self):
        for payload in ([], [{"icaoId": "ZUCK"}], [{"icaoId": "ZUCK", "temp": None}]):
            with self.subTest(payload=payload), patch.object(
                data_source.requests, "get"
            ) as request:
                response = request.return_value
                response.status_code = 200
                response.json.return_value = payload

                with self.assertRaises(RuntimeError):
                    data_source.aviationweather_temperature("ZUCK")

    def test_ensemble_forecast_requests_celsius(self):
        payload = {
            "latitude": 29.75,
            "longitude": 106.75,
            "hourly": {"time": [123456], "temperature_2m": [31.0]},
        }
        with patch.object(data_source.requests, "get") as request:
            request.return_value.json.return_value = payload

            result = data_source.ensemble_forcast()

        self.assertEqual(result["data"]["temperature_2m"].iloc[0], 31.0)
        self.assertEqual(
            request.call_args.kwargs["params"]["temperature_unit"], "celsius"
        )


class RealTimeDataTest(unittest.TestCase):
    def setUp(self):
        real_time_data.temperature_history.clear()
        self.addCleanup(real_time_data.temperature_history.clear)

    def test_update_temperature_appends_aviationweather_celsius_temperature(self):
        city_config = [
            {
                "name": "TestCity",
                "ICAO": "ZUCK",
            },
            {
                "name": "TestCityCenter",
                "ICAO": "ZUCK",
            },
        ]

        with (
            patch.object(real_time_data.config, "CITY", city_config),
            patch.object(
                real_time_data, "aviationweather_temperature", return_value=31.0
            ) as aviationweather,
            patch.object(real_time_data.time, "time", return_value=654321),
        ):
            updated, new_temperatures = real_time_data.update_temperature()

        temps = real_time_data.temperature_history["TestCity"]
        center_temps = real_time_data.temperature_history["TestCityCenter"]
        self.assertTrue(updated)
        self.assertEqual(len(new_temperatures), 2)
        self.assertEqual(list(temps.columns), ["update_time", "temperature"])
        self.assertEqual(
            temps.iloc[0].to_dict(),
            {"update_time": 654321.0, "temperature": 31.0},
        )
        self.assertEqual(center_temps.iloc[0]["temperature"], 31.0)
        aviationweather.assert_called_once_with("ZUCK")

if __name__ == "__main__":
    unittest.main()
