from __future__ import annotations

import argparse
import json
import pathlib
import joblib

from .features import extract_features


DEFAULT_MODEL_PATH = pathlib.Path("models/security_classifier.joblib")


def load_model(model_path: pathlib.Path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["vectorizer"], bundle.get("features", [])


def classify_snippet(code: str, model_path: pathlib.Path = DEFAULT_MODEL_PATH, language: str | None = None) -> dict:
    model, vectorizer, names = load_model(model_path)
    feats = extract_features(code, language=language)
    vector = vectorizer.transform([feats])
    proba = model.predict_proba(vector)[0][1]
    label = "VULNERABLE" if proba >= 0.5 else "SEGURO"
    return {"label": label, "probabilidad_vulnerable": float(proba), "features": feats, "campos": names}


def main():
    parser = argparse.ArgumentParser(description="Clasifica un fragmento de código con el modelo entrenado")
    parser.add_argument("code_path", type=pathlib.Path, help="Ruta al archivo de código a evaluar")
    parser.add_argument("--language", type=str, default=None, help="Idioma del código (ej. python, c, cpp)")
    parser.add_argument("--model-path", type=pathlib.Path, default=DEFAULT_MODEL_PATH, help="Ruta al modelo .joblib")
    args = parser.parse_args()

    code = args.code_path.read_text(encoding="utf-8")
    result = classify_snippet(code, args.model_path, language=args.language)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
