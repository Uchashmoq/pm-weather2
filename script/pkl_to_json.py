import argparse
import json
import pickle
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return [
                {str(k): _to_jsonable(v) for k, v in row.items()}
                for row in value.reset_index().to_dict(orient="records")
            ]
        if isinstance(value, pd.Series):
            return {
                str(k): _to_jsonable(v)
                for k, v in value.to_dict().items()
            }
        if isinstance(value, pd.Index):
            return [_to_jsonable(item) for item in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if value is pd.NA:
            return None
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=repr)]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return repr(value)


def convert_pkl_to_json(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = input_path.with_suffix(".json")

    with input_path.open("rb") as f:
        data = pickle.load(f)

    json_data = _to_jsonable(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a pickle file to readable JSON.")
    parser.add_argument("input", help="Path to the input .pkl file")
    parser.add_argument("-o", "--output", help="Path to the output .json file")
    args = parser.parse_args()

    output_path = convert_pkl_to_json(
        Path(args.input),
        Path(args.output) if args.output else None,
    )
    print(output_path)


if __name__ == "__main__":
    main()
