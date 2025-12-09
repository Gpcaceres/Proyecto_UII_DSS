from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional

import pandas as pd

DEFAULT_RAW_PATH = pathlib.Path("data/MSR_data_cleaned.csv")
DEFAULT_OUTPUT_PATH = pathlib.Path("data/bigvul_pipeline.csv")


def _normalize_language(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    lang = value.strip().lower()
    mapping = {
        "c++": "cpp",
        "c/c++": "cpp",
        "cpp": "cpp",
    }
    return mapping.get(lang, lang or "unknown")


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if pd.isna(value) else str(value)
    return str(value)


def _build_records(df: pd.DataFrame) -> List[dict]:
    records: List[dict] = []
    for idx, row in df.iterrows():
        lang = _normalize_language(_safe_str(row.get("lang")))
        commit_id = _safe_str(row.get("commit_id")) or f"row{idx}"
        base_id = f"{commit_id}_{idx}"

        vul_flag = _safe_str(row.get("vul"))
        is_vulnerable = vul_flag not in {"", "0", "false", "False"}

        func_before = _safe_str(row.get("func_before")).strip()
        func_after = _safe_str(row.get("func_after")).strip()

        if func_before:
            records.append(
                {
                    "id": f"{base_id}_before",
                    "label": "vulnerable" if is_vulnerable else "seguro",
                    "language": lang,
                    "code": func_before,
                }
            )

        if func_after:
            records.append(
                {
                    "id": f"{base_id}_after",
                    "label": "seguro",
                    "language": lang,
                    "code": func_after,
                }
            )
    return records


def convert_dataset(raw_path: pathlib.Path = DEFAULT_RAW_PATH, output_path: pathlib.Path = DEFAULT_OUTPUT_PATH) -> pathlib.Path:
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset original en {raw_path}")

    df = pd.read_csv(raw_path)
    records = _build_records(df)
    pipeline_df = pd.DataFrame(records, columns=["id", "label", "language", "code"])
    pipeline_df = pipeline_df[pipeline_df["code"].str.len() > 0]
    pipeline_df = pipeline_df.drop_duplicates(subset=["code", "label"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_df.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convierte MSR_data_cleaned.csv al formato del pipeline")
    parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_RAW_PATH, help="Ruta al CSV original")
    parser.add_argument(
        "--output", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH, help="Ruta de salida para el CSV convertido"
    )
    args = parser.parse_args()

    output = convert_dataset(args.input, args.output)
    print(f"Dataset convertido guardado en {output}")


if __name__ == "__main__":
    main()
