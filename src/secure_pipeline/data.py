from __future__ import annotations

import pathlib
import textwrap
from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd
from urllib import request

# Esta URL solo se usa si NO pasas --dataset en la línea de comandos.
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/MLWhiz/data_vuln_demo/main/demo_dataset.csv"


@dataclass
class DatasetConfig:
    url: str = DEFAULT_DATA_URL
    # Preferimos un dataset local en formato JSON (si existe)
    local_path: pathlib.Path = pathlib.Path("data/MSR_data_cleaned.json")

    def ensure(self) -> pathlib.Path:
        """Devuelve la ruta del dataset.
        Busca varios formatos que pueda haber en la carpeta data:
        1) la ruta preferida (JSON),
        2) archivos CSV conocidos (MSR_data_cleaned.csv, demo_dataset.csv),
        Si no hay ninguno disponible, descarga el dataset por defecto desde self.url
        y lo guarda con la extensión indicada por la URL.
        """
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        # Si existe la ruta preferida, úsala
        if self.local_path.exists():
            return self.local_path

        # Comprobar rutas alternativas
        alternatives = [
            self.local_path.with_suffix(".csv"),
            pathlib.Path("data/demo_dataset.csv"),
            pathlib.Path("data/bigvul_pipeline.csv"),
        ]
        for alt in alternatives:
            if alt.exists():
                return alt

        # Si no existe nada, descargar desde URL al sufijo adecuado (según URL)
        suffix = pathlib.Path(self.url).suffix or ".csv"
        download_path = self.local_path.with_suffix(suffix)
        with request.urlopen(self.url, timeout=30) as response:
            content = response.read().decode("utf-8")
        download_path.write_text(content, encoding="utf-8")
        return download_path


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Carga datasets en formato CSV, JSON o JSONL con columnas: id, label, language, code."""
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        # Intentar leer como JSONL (líneas JSON) primero, si falla, intentar JSON normal
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path, lines=False)
    else:
        df = pd.read_csv(path)

    # Asegurarnos de que exista una columna 'code'; intentar alternativas comunes o
    # reconocer formatos MSR (func_before/func_after) y convertirlos al esquema del
    # pipeline (id,label,language,code)
    if "code" not in df.columns:
        if "content" in df.columns:
            df["code"] = df["content"]
        elif "code_text" in df.columns:
            df["code"] = df["code_text"]
        elif "snippet" in df.columns:
            df["code"] = df["snippet"]
        elif ("func_before" in df.columns) or ("func_after" in df.columns):
            # Convertir formato MSR a filas por función (similar a convert_bigvul._build_records)
            records = []
            for idx, row in df.iterrows():
                lang = (row.get("lang") or row.get("language") or "unknown")
                commit_id = str(row.get("commit_id") or f"row{idx}")
                base_id = f"{commit_id}_{idx}"
                vul_flag = str(row.get("vul") or row.get("vulnerable") or "")
                is_vulnerable = vul_flag not in {"", "0", "false", "False"}

                func_before = row.get("func_before") or ""
                func_after = row.get("func_after") or ""

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
            df = pd.DataFrame(records)
        else:
            raise ValueError("Dataset missing required column: 'code'")

    expected_cols = {"id", "label", "language", "code"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Normaliza etiquetas a minúsculas, por si acaso
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


def summarize_dataset(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts()
    return textwrap.dedent(
        f"""
        Registros: {len(df)}
        Clases: {counts.to_dict()}
        Lenguajes: {sorted(df['language'].unique())}
        """
    ).strip()


def iter_samples(df: pd.DataFrame) -> Iterable[Tuple[str, str, str]]:
    """
    Devuelve (code, label, language) para cada fila,
    tal como espera train.py
    """
    for _, row in df.iterrows():
        code = row["code"]
        label = row["label"]
        language = row.get("language", "unknown")
        yield code, label, language
