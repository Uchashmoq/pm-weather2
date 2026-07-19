# Polymarket Weather Tool 2
A tool to calculate the probability of highest temperatures.
Real-time temperature data: AviationWeather METAR API.
Weather forecast API: Open-Meteo ensemble API.
All temperatures in the project are Celsius.

## Init
```bash
python3 -m venv venv
source venv/bin/activate
pip install requests pandas numpy fastapi "uvicorn[standard]" jinja2
```

## Run
```bash
python main.py
python main.py --web
python main.py --web --web-host 0.0.0.0 --web-port 8000
```

When `--web` is enabled, reports are shown at `http://127.0.0.1:8000`.

## Example Data
### aviation weather
`https://aviationweather.gov/api/data/metar?ids=ZUCK&format=json`

The METAR `temp` field is the observed air temperature in Celsius. Each city is
mapped to a station with its `ICAO` value in `config.py`.

```json
[
  {
    "icaoId": "ZUCK",
    "receiptTime": "2026-07-19T14:05:28.124Z",
    "obsTime": 1784469600,
    "reportTime": "2026-07-19T14:00:00.000Z",
    "temp": 31,
    "dewp": 21,
    "wdir": 100,
    "wspd": 4,
    "visib": "6+",
    "altim": 1004,
    "qcField": 16,
    "metarType": "METAR",
    "rawOb": "METAR ZUCK 191400Z 10002MPS CAVOK 31/21 Q1004 NOSIG",
    "lat": 29.718,
    "lon": 106.639,
    "elev": 416,
    "name": "Chongqing/Jiangbei Intl, CQ, CN",
    "cover": "CAVOK",
    "clouds": [],
    "fltCat": "VFR"
  }
]
```
