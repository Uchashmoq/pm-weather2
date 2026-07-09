# Polymarket Weather Tool 2
A tool to cauculate the probability of highest temperatures.
Real-time temperature data crawled from Wunderground.
Weather forecast API: open-meteo.com

## Init
```bash
python3 -m venv venv
source venv/bin/activate
pip install requests pandas numpy fastapi "uvicorn[standard]" jinja2
```

## Run
```bash
python3 main.py
python3 main.py --web
python3 main.py --web --web-host 0.0.0.0 --web-port 8000
```

When `--web` is enabled, reports are shown at `http://127.0.0.1:8000`.
