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
    # Ruta por defecto cuando se descarga automáticamentes
    local_path: pathlib.Path = pathlib.Path("data/demo_dataset.csv")

    def ensure(self) -> pathlib.Path:
        """Devuelve la ruta del dataset.
        Si el archivo ya existe en local, NO descarga nada.
        Si no existe, intenta descargarlo desde self.url.
        """
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.local_path.exists():
            return self.local_path

        with request.urlopen(self.url, timeout=30) as response:
            content = response.read().decode("utf-8")

        self.local_path.write_text(content, encoding="utf-8")
        return self.local_path


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Carga cualquier CSV que tenga las columnas: id, label, language, code."""
    df = pd.read_csv(path)
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
