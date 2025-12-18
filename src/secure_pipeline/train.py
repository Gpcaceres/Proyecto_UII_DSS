from __future__ import annotations

import argparse
import pathlib
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from .data import DatasetConfig, iter_samples, load_dataset, summarize_dataset
from .features import extract_features, feature_names


MODEL_DEFAULT_PATH = pathlib.Path("models/security_classifier.joblib")


# -----------------------------------------------------------
# Vectorizador
# -----------------------------------------------------------
def build_vectorizer():
    return DictVectorizer(sparse=False)


def prepare_features(codes, languages=None):
    languages = languages or [None] * len(codes)
    return [extract_features(code, lang) for code, lang in zip(codes, languages)]


# -----------------------------------------------------------
# Entrenamiento principal
# -----------------------------------------------------------
def train_model(dataset_path: pathlib.Path, model_path: pathlib.Path) -> None:
    df = load_dataset(dataset_path)
    print("Dataset listo:\n" + summarize_dataset(df))

    codes, labels, languages = zip(*iter_samples(df))

    # Features
    X_dicts = prepare_features(codes, languages)
    y = np.array([1 if label == "vulnerable" else 0 for label in labels])

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(X_dicts)

    # -----------------------------------------------------------
    # Modelo optimizado para desbalance severo
    # -----------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    # -----------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------
    class_counts = np.bincount(y)
    min_class = class_counts.min() if len(class_counts) else 0
    cv = min(5, len(y), min_class)

    if cv >= 2:
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        print(f"\n📌 Validación cruzada ({cv}-fold):")
        print(f" - Accuracy media : {acc_scores.mean():.3f} (+/- {acc_scores.std():.3f})")
        print(f" - F1-score media : {f1_scores.mean():.3f} (+/- {f1_scores.std():.3f})\n")
    else:
        print("Dataset demasiado pequeño para validación cruzada; entrenando sin CV")

    # -----------------------------------------------------------
    # Entrenamiento final
    # -----------------------------------------------------------
    model.fit(X, y)
    preds = model.predict(X)

    print("\n📌 Reporte de clasificación (sobre dataset completo):")
    print(classification_report(y, preds, target_names=["seguro", "vulnerable"]))

    # Guardado del modelo
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "features": feature_names()},
        model_path
    )

    print(f"Modelo guardado en {model_path}")


# -----------------------------------------------------------
# Punto de entrada CLI
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrena un clasificador de vulnerabilidades")
    parser.add_argument("--dataset", type=pathlib.Path, default=DatasetConfig().local_path,
                        help="Ruta al dataset (CSV o JSON). Por defecto intentará usar data/MSR_data_cleaned.json si está presente")
    parser.add_argument("--model-path", type=pathlib.Path, default=MODEL_DEFAULT_PATH,
                        help="Ruta de salida del modelo .joblib")
    args = parser.parse_args()

    config = DatasetConfig(local_path=args.dataset)
    dataset_path = config.ensure()
    train_model(dataset_path, args.model_path)


if __name__ == "__main__":
    main()
