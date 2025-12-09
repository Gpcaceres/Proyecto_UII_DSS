from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Dict

import joblib

from .features import extract_features

MODEL_PATH = pathlib.Path("models/security_classifier.joblib")


def load_model():
    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encontró el modelo en {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    vectorizer = data["vectorizer"]
    return model, vectorizer


def is_code_file(path: pathlib.Path) -> bool:
    return path.suffix.lower() in {".c", ".cpp", ".h", ".hpp"}


def build_dataset(files: List[pathlib.Path]) -> List[Dict]:
    records = []
    for f in files:
        if not f.exists():
            # Puede pasar si el archivo se borró en el PR
            continue
        if not is_code_file(f):
            continue
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] No se pudo leer {f}: {e}", file=sys.stderr)
            continue

        features = extract_features(source, lang=f.suffix.lstrip("."))
        records.append({"path": str(f), "features": features})
    return records


def run_check(files: List[str]) -> int:
    # Filtrar lista vacía
    paths = [pathlib.Path(f) for f in files if f.strip()]
    if not paths:
        print("No hay archivos relevantes para analizar. Aprobando por defecto.")
        return 0

    records = build_dataset(paths)
    if not records:
        print("No se encontraron archivos de código C/C++ en el diff. Aprobando.")
        return 0

    model, vectorizer = load_model()

    X = vectorizer.transform([r["features"] for r in records])
    proba = model.predict_proba(X)  # columnas: [P(seguro), P(vulnerable)]
    preds = model.predict(X)        # 0 = seguro, 1 = vulnerable

    summary = []
    has_vulnerable = False

    for r, y_hat, p in zip(records, preds, proba):
        prob_vul = float(p[1])
        label = "vulnerable" if y_hat == 1 else "seguro"
        if label == "vulnerable":
            has_vulnerable = True

        summary.append(
            {
                "path": r["path"],
                "label": label,
                "probabilidad_vulnerable": prob_vul,
            }
        )

    result = {
        "resultado_global": "VULNERABLE" if has_vulnerable else "SEGURO",
        "archivos_analizados": len(records),
        "detalles": summary,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Código de salida: 1 si hay vulnerable → falla el job
    return 1 if has_vulnerable else 0


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el clasificador sobre los archivos modificados en un PR."
    )
    parser.add_argument("files", nargs="*", help="Lista de archivos a analizar")
    args = parser.parse_args()

    exit_code = run_check(args.files)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
