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

    # Load dataset
    dataset_path = args.dataset
    print('Cargando dataset desde', dataset_path)
    df = load_dataset(dataset_path)
    print(summarize_dataset(df))

    # Train/test split (hold-out test)
    train_df, test_df = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=args.random_state)
    print('Train:', len(train_df), 'Test:', len(test_df))

    # Sample for hyperparameter tuning
    sample_df = sample_dataframe(train_df, max_samples=args.sample_size, random_state=args.random_state)
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
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)

    search = RandomizedSearchCV(model, param_dist, n_iter=args.n_iter, scoring='accuracy', cv=cv, verbose=2, random_state=args.random_state, n_jobs=-1)
    print('Ejecutando búsqueda...')
    search.fit(X_sample, y_sample)

    print('\nMEJORES PARÁMETROS:')
    print(pformat(search.best_params_))
    print('MEJOR SCORE (CV):', search.best_score_)

    # Entrenar modelo con mejores parámetros sobre todo el train_df y evaluar en test_df
    print('\nVectorizando entrenamiento completo (train_df)')
    codes_train, labels_train, languages_train = zip(*[(r['code'], r['label'], r.get('language', None)) for _, r in train_df.iterrows()])
    X_train_dicts = prepare_features(codes_train, languages_train)
    vec_full = DictVectorizer(sparse=False)
    X_train = vec_full.fit_transform(X_train_dicts)
    y_train = np.array([1 if l == 'vulnerable' else 0 for l in labels_train])

    best_params = search.best_params_
    best_model = RandomForestClassifier(**{k: v for k, v in best_params.items() if k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features','class_weight','criterion']}, random_state=args.random_state, n_jobs=-1)
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
    print(classification_report(y_test, preds, target_names=['seguro','vulnerable']))
    print('Accuracy test:', accuracy_score(y_test, preds))

    # Guardar modelo + vectorizer
    out = args.out_model
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': best_model, 'vectorizer': vec_full, 'features': feature_names()}, out)
    print('Modelo guardado en', out)


if __name__ == '__main__':
    main()
