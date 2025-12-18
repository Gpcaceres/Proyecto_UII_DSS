from __future__ import annotations

import argparse
import pathlib
import time
from pprint import pformat

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score

from secure_pipeline.data import load_dataset, summarize_dataset
from secure_pipeline.train import prepare_features, build_vectorizer
from secure_pipeline.features import feature_names


def sample_dataframe(df, max_samples: int = 100_000, random_state: int = 42):
    # Stratified sampling by label to keep class distribution
    if len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    return df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))


def vectorize_features(codes, languages):
    X_dicts = prepare_features(codes, languages)
    vec = build_vectorizer()
    X = vec.fit_transform(X_dicts)
    return X, vec


def main():
    parser = argparse.ArgumentParser(description='Optimize RandomForest on dataset (sample and tune)')
    parser.add_argument('--dataset', type=pathlib.Path, default=pathlib.Path('data/bigvul_pipeline.csv'), help='CSV/JSON pipeline dataset')
    parser.add_argument('--sample-size', type=int, default=100000, help='Number of samples to use for hyperparameter search')
    parser.add_argument('--n-iter', type=int, default=40, help='Number of iterations for RandomizedSearchCV')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out-model', type=pathlib.Path, default=pathlib.Path('models/security_classifier_opt.joblib'))
    args = parser.parse_args()

    # Load or prepare dataset: support large JSON by converting to pipeline CSV in streaming mode
    dataset_path = args.dataset
    print('Preparando dataset desde', dataset_path)
    if dataset_path.suffix.lower() in {'.json', '.jsonl'}:
        print('Archivo JSON grande detectado — generando CSV pipeline en streaming (puede tardar)')
        from secure_pipeline.convert_bigvul import convert_dataset
        dataset_path = convert_dataset(dataset_path, pathlib.Path('data/bigvul_pipeline.csv'))
        print('CSV generado en', dataset_path)

    # Para datasets grandes, muestreamos un sample con reservoir sampling desde el CSV
    def sample_from_csv(path: pathlib.Path, sample_size: int, random_state: int = 42):
        import csv, random, sys
        # Increase field size limit to handle large code fields
        try:
            csv.field_size_limit(10 * 1024 * 1024)
        except Exception:
            try:
                csv.field_size_limit(sys.maxsize)
            except Exception:
                pass
        rnd = random.Random(random_state)
        sample = []
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            i = 0
            for row in reader:
                i += 1
                try:
                    # Ensure required fields exist; skip malformed rows
                    if 'code' not in row:
                        continue
                    if i <= sample_size:
                        sample.append(row)
                    else:
                        j = rnd.randrange(i)
                        if j < sample_size:
                            sample[j] = row
                except csv.Error:
                    # Skip rows that exceed parser limits even after increasing field size
                    continue
        import pandas as pd
        return pd.DataFrame(sample)

    # Train/test split (hold-out test) — for large datasets we sample first, then split
    df_sample = sample_from_csv(dataset_path, max(1000, args.sample_size), random_state=args.random_state)
    print('Muestra cargada para pre-evaluación:', len(df_sample))

    # Ensure label normalization like load_dataset
    df_sample['label'] = df_sample['label'].astype(str).str.lower().str.strip()

    try:
        train_df, test_df = train_test_split(df_sample, test_size=0.10, stratify=df_sample['label'], random_state=args.random_state)
    except ValueError:
        train_df, test_df = train_test_split(df_sample, test_size=0.10, random_state=args.random_state)
    print('Train (muestra):', len(train_df), 'Test (muestra):', len(test_df))

    # Sample for hyperparameter tuning from the full CSV via reservoir sampling on the train split
    # We'll sample codes from the original full CSV, but respecting label distribution from train_df
    # For simplicity, just take a combined random sample of size args.sample_size from CSV
    sample_df = sample_from_csv(dataset_path, args.sample_size, random_state=args.random_state)
    print('Sample para búsqueda:', len(sample_df))

    codes, labels, languages = zip(*[(r['code'], r['label'], r.get('language', None)) for _, r in sample_df.iterrows()])
    X_sample, vec = vectorize_features(codes, languages)
    y_sample = np.array([1 if l == 'vulnerable' else 0 for l in labels])

    # Define param distributions
    param_dist = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'criterion': ['gini', 'entropy']
    }

    model = RandomForestClassifier(random_state=args.random_state, n_jobs=-1)
    # Adjust CV folds to the sample size
    n_splits = min(3, max(2, len(y_sample)))
    if len(y_sample) < 4:
        # Too small for meaningful CV — fallback to simple fit on sample
        print('Muestra muy pequeña para búsqueda con CV; entrenando sin búsqueda hiperparámetros')
        X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.25, random_state=args.random_state)
        model.set_params(**{'n_estimators': 600, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample'})
        model.fit(X_train, y_train)
        class BestLike:
            best_params_ = model.get_params()
            best_score_ = None
        search = BestLike()
        best_model = model
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
        search = RandomizedSearchCV(model, param_dist, n_iter=args.n_iter, scoring='accuracy', cv=cv, verbose=2, random_state=args.random_state, n_jobs=-1)
        print('Ejecutando búsqueda...')
        search.fit(X_sample, y_sample)
        best_model = search.best_estimator_

    print('\nMEJORES PARÁMETROS:')
    # search may be a simple BestLike object (for tiny samples)
    best_params = getattr(search, 'best_params_', getattr(search, 'best_params_', None))
    try:
        print(pformat(search.best_params_))
    except Exception:
        print('No hay best_params_ (ejecución reducida)')
    try:
        print('MEJOR SCORE (CV):', search.best_score_)
    except Exception:
        print('MEJOR SCORE (CV): None')

    # Guardar artefactos de la búsqueda
    import os
    import datetime
    exp_dir = pathlib.Path('models/experiments') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Guardar cv_results_ si existe
        if hasattr(search, 'cv_results_'):
            joblib.dump(search.cv_results_, exp_dir / 'cv_results.joblib')
        if hasattr(search, 'best_params_'):
            joblib.dump(search.best_params_, exp_dir / 'best_params.joblib')
        # Guardar search object para reproducibilidad
        joblib.dump(search, exp_dir / 'search_full.joblib')
        print('Artefactos guardados en', exp_dir)
    except Exception as e:
        print('No fue posible guardar artefactos de búsqueda:', e)

    # Entrenar modelo con mejores parámetros sobre todo el train_df y evaluar en test_df
    print('\nVectorizando entrenamiento completo (train_df)')
    codes_train, labels_train, languages_train = zip(*[(r['code'], r['label'], r.get('language', None)) for _, r in train_df.iterrows()])
    X_train_dicts = prepare_features(codes_train, languages_train)
    vec_full = DictVectorizer(sparse=False)
    X_train = vec_full.fit_transform(X_train_dicts)
    y_train = np.array([1 if l == 'vulnerable' else 0 for l in labels_train])

    # Build best_model with available params
    if hasattr(search, 'best_params_'):
        params = {k: v for k, v in search.best_params_.items() if k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features','class_weight','criterion']}
    else:
        params = {'n_estimators': 600, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample'}

    best_model = RandomForestClassifier(**params, random_state=args.random_state, n_jobs=-1)
    print('Entrenando modelo final con mejores parámetros sobre todo el train...')
    t0 = time.time()
    best_model.fit(X_train, y_train)
    print('Tiempo entrenamiento (s):', time.time() - t0)

    # Evaluar en set de test
    print('Vectorizando y evaluando en set de test...')
    codes_test, labels_test, languages_test = zip(*[(r['code'], r['label'], r.get('language', None)) for _, r in test_df.iterrows()])
    X_test_dicts = prepare_features(codes_test, languages_test)
    X_test = vec_full.transform(X_test_dicts)
    y_test = np.array([1 if l == 'vulnerable' else 0 for l in labels_test])

    preds = best_model.predict(X_test)
    print('\nReporte final (TEST):')
    # Handle cases where test set contains only one class
    labels_unique = np.unique(y_test)
    try:
        if len(labels_unique) == 1:
            # Map label to name
            name = 'vulnerable' if labels_unique[0] == 1 else 'seguro'
            print(f"Test contiene una sola clase ({name}); mostrando métricas básicas.")
            print('Accuracy test:', accuracy_score(y_test, preds))
        else:
            print(classification_report(y_test, preds, target_names=['seguro','vulnerable']))
            print('Accuracy test:', accuracy_score(y_test, preds))
    except Exception as e:
        print('No fue posible generar classification_report:', e)
        print('Accuracy test:', accuracy_score(y_test, preds))

    # Guardar modelo + vectorizer
    out = args.out_model
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': best_model, 'vectorizer': vec_full, 'features': feature_names()}, out)
    print('Modelo guardado en', out)


if __name__ == '__main__':
    main()
