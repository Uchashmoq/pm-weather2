import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
REPORT_JSONL_FILE = BASE_DIR / "data" / "report" / "report.jsonl"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Weather Reports")


def report_key(report):
    content = {k: v for k, v in report.items() if k != "created_at"}
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def add_diff_marks(report):
    calibrated = report.get("calibrated")
    if not calibrated:
        report["diff_cells"] = []
        return report

    raw_rows = report["raw"]["rows"]
    calibrated_rows = calibrated["rows"]
    diff_cells = []
    for row_idx, (raw_row, calibrated_row) in enumerate(zip(raw_rows, calibrated_rows)):
        for col_idx, (raw_cell, calibrated_cell) in enumerate(
            zip(raw_row, calibrated_row)
        ):
            if col_idx > 0 and raw_cell != calibrated_cell:
                diff_cells.append(f"{row_idx}:{col_idx}")

    report["diff_cells"] = diff_cells
    return report


def load_reports(limit: int = 100):
    if not REPORT_JSONL_FILE.exists():
        return []

    reports = []
    with REPORT_JSONL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    deduped = {}
    for report in reports:
        deduped[report_key(report)] = report

    return [add_diff_marks(report) for report in reversed(list(deduped.values())[-limit:])]



@app.get("/")
async def reports_page(request: Request, limit: int = 50):
    reports = load_reports(limit)
    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            "reports": reports,
            "limit": limit,
            "report_file": str(REPORT_JSONL_FILE.relative_to(BASE_DIR)),
        },
    )


@app.get("/api/reports")
async def reports_api(limit: int = 100):
    return JSONResponse({"reports": load_reports(limit)})
