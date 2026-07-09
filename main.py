import config
import forecast
from forecast import forecast_history
from real_time_data import temperature_history
import predict
import real_time_data
import argparse
import asyncio
import threading
import json
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

REPORT_DIR = Path(__file__).resolve().parent / "data" / "report"
REPORT_FILE = REPORT_DIR / "report.txt"
REPORT_JSONL_FILE = REPORT_DIR / "report.jsonl"


def predict_to_str(predict, calibrated) -> str:
    report: pd.DataFrame = predict["report"]
    update_time = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(predict["update_time"])
    )
    table = report.copy()
    for col in table.columns:
        if col != "date":
            table[col] = table[col].map(lambda prob: f"{prob:.2%}")
    return "\n".join(
        [
            f'city={predict["city"]} model={predict["model"]} calibrated={calibrated} forecast_update_time={update_time}',
            table.to_string(index=False),
        ]
    )


def print_and_append_report(ps: str):
    print(ps)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("a", encoding="utf-8") as f:
        print(ps, file=f)
        print(file=f)


def prediction_table_to_json(prediction):
    report: pd.DataFrame = prediction["report"]
    table = report.copy()
    for col in table.columns:
        if col != "date":
            table[col] = table[col].map(lambda prob: f"{prob:.2%}")

    return {
        "columns": list(table.columns),
        "rows": table.values.tolist(),
    }


def append_report_jsonl(raw_prediction, calibrated_prediction=None):
    update_time = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(raw_prediction["update_time"])
    )
    record = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "city": raw_prediction["city"],
        "model": raw_prediction["model"],
        "forecast_update_time": update_time,
        "raw": prediction_table_to_json(raw_prediction),
        "calibrated": (
            prediction_table_to_json(calibrated_prediction)
            if calibrated_prediction is not None
            else None
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_JSONL_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def predicting():
    while True:
        for city in config.CITY:
            city_name = city["name"]
            if city_name in forecast_history:
                for model_name, li in forecast_history[city_name].items():
                    if len(li) > 0:
                        fc = li[-1]
                        res = predict.predict_city(city, fc, correction=False)
                        raw_prediction = {
                            "city": city_name,
                            "model": model_name,
                            "update_time": fc["update_time"],
                            "report": res,
                        }
                        ps = predict_to_str(raw_prediction, calibrated=False)
                        print_and_append_report(ps)
                        calibrated_prediction = None
                        if city_name in temperature_history:
                            actual_temps = temperature_history[city_name]
                            if len(actual_temps) > 0:
                                res_calibrated = predict.predict_city(
                                    city, fc, actual_temps, correction=True
                                )
                                calibrated_prediction = {
                                    "city": city_name,
                                    "model": model_name,
                                    "update_time": fc["update_time"],
                                    "report": res_calibrated,
                                }
                                ps = predict_to_str(
                                    calibrated_prediction, calibrated=True
                                )
                                print_and_append_report(ps)
                        append_report_jsonl(raw_prediction, calibrated_prediction)

        await asyncio.sleep(config.UPDATE_PREDICTION_INTERVAL)


async def run_web_server(host: str, port: int):
    import uvicorn
    from web import app

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def main(
    start_web: bool = False,
    web_host: str = "127.0.0.1",
    web_port: int = 8000,
):
    tasks = [
        asyncio.create_task(forecast.update_forecast_periotically()),
        asyncio.create_task(real_time_data.update_temperature_periotically()),
    ]
    if start_web:
        tasks.append(asyncio.create_task(run_web_server(web_host, web_port)))  # type: ignore
        logging.info("web server starting at http://%s:%d", web_host, web_port)

    await asyncio.sleep(4)
    logging.info("starting ... ")
    tasks.append(asyncio.create_task(predicting()))

    await asyncio.gather(*tasks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="start report web server")
    parser.add_argument("--web-host", default="127.0.0.1", help="web server host")
    parser.add_argument("--web-port", type=int, default=8000, help="web server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.web, args.web_host, args.web_port))
