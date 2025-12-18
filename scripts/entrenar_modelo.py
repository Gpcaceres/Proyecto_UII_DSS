"""
Script completo de entrenamiento con métricas detalladas.

Este script:
1. Carga el dataset procesado (msr_pipeline.csv o MSR_data_cleaned.csv)
2. Extrae features del código
3. Entrena un modelo RandomForest optimizado para desbalance
4. Genera métricas completas de evaluación
5. Guarda el modelo entrenado
6. Genera reportes de resultados
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para Windows
import matplotlib.pyplot as plt
import seaborn as sns


# Importar desde el módulo del pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from secure_pipeline.data import load_dataset
from secure_pipeline.features import extract_features
from sklearn.feature_extraction import DictVectorizer


def train_model_complete(dataset_path: str, 
                        model_path: str = "models/security_classifier_msr.joblib",
                        test_size: float = 0.2,
                        generate_plots: bool = True):
    """
    Entrenamiento completo con validación y métricas detalladas.
    """
    
    print("🚀 Entrenamiento del Modelo de Seguridad")
    print("=" * 70)
    
    # 1. Cargar dataset
    print(f"\n📂 Cargando dataset: {dataset_path}")
    df = load_dataset(Path(dataset_path))
    print(f"✅ Dataset cargado: {len(df)} registros")
    
    # Mostrar distribución
    print("\n📊 Distribución de clases:")
    for label, count in df['label'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  - {label}: {count} ({pct:.2f}%)")
    
    # 2. Preparar datos
    print("\n🔧 Extrayendo features del código...")
    
    X_dicts = []
    y = []
    
    for idx, row in df.iterrows():
        code = row['code']
        label = row['label']
        language = row.get('language', 'C')
        
        # Extraer features
        features = extract_features(code, language)
        X_dicts.append(features)
        
        # Etiqueta binaria
        y.append(1 if label == 'vulnerable' else 0)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Procesadas {idx + 1}/{len(df)} muestras...")
    
    y = np.array(y)
    
    # Vectorizar features
    print("\n📐 Vectorizando features...")
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X_dicts)
    
    print(f"✅ Features extraídas: {X.shape[1]} dimensiones")
    print(f"✅ Muestras totales: {X.shape[0]}")
    
    # 3. División train/test
    print(f"\n✂️  Dividiendo dataset (test={test_size*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Test:  {len(X_test)} muestras")
    
    # 4. Entrenar modelo
    print("\n🤖 Entrenando modelo RandomForest...")
    print("   Configuración: 600 árboles, max_depth=20, class_weight=balanced_subsample")
    
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("✅ Modelo entrenado")
    
    # 5. Validación cruzada (en train)
    print("\n📊 Validación cruzada (5-fold)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    print(f"  F1-Score CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # 6. Evaluación en test
    print("\n🎯 Evaluación en conjunto de prueba...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas principales
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_roc = None
    
    print(f"\n📈 Resultados en Test:")
    print(f"  - Accuracy:  {accuracy:.3f}")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")
    if auc_roc:
        print(f"  - AUC-ROC:   {auc_roc:.3f}")
    
    # Reporte detallado
    print(f"\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, 
                               target_names=['seguro', 'vulnerable'],
                               digits=3))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"🔢 Matriz de confusión:")
    print(f"                Predicho Seguro  Predicho Vulnerable")
    print(f"Real Seguro          {cm[0][0]:6d}            {cm[0][1]:6d}")
    print(f"Real Vulnerable      {cm[1][0]:6d}            {cm[1][1]:6d}")
    
    # 7. Feature importance
    print(f"\n⭐ Top 10 features más importantes:")
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # 8. Guardar modelo
    print(f"\n💾 Guardando modelo...")
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        'model': model,
        'vectorizer': vectorizer,
        'feature_names': list(feature_names),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        },
        'trained_at': datetime.now().isoformat(),
        'dataset': dataset_path,
        'n_samples': len(df)
    }, model_output)
    
    print(f"✅ Modelo guardado en: {model_path}")
    
    # 9. Generar gráficos
    if generate_plots:
        print(f"\n📊 Generando visualizaciones...")
        try:
            plot_dir = Path("models/plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Gráfico 1: Matriz de confusión
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Seguro', 'Vulnerable'],
                       yticklabels=['Seguro', 'Vulnerable'])
            plt.title('Matriz de Confusión')
            plt.ylabel('Clase Real')
            plt.xlabel('Clase Predicha')
            plt.tight_layout()
            plt.savefig(plot_dir / 'confusion_matrix.png', dpi=150)
            plt.close()
            
            # Gráfico 2: Feature importance
            plt.figure(figsize=(10, 6))
            top_n = 20
            indices = np.argsort(importances)[::-1][:top_n]
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importancia')
            plt.title('Top 20 Features Más Importantes')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plot_dir / 'feature_importance.png', dpi=150)
            plt.close()
            
            print(f"✅ Gráficos guardados en: {plot_dir}")
        except Exception as e:
            print(f"⚠️  Error al generar gráficos: {e}")
    
    # 10. Generar reporte
    print(f"\n📄 Generando reporte de entrenamiento...")
    report_path = Path("models/training_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - MODELO DE SEGURIDAD\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Modelo: {model_path}\n\n")
        
        f.write("--- Dataset ---\n")
        f.write(f"Total de muestras: {len(df)}\n")
        f.write(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
        f.write(f"Features: {X.shape[1]}\n\n")
        
        f.write("--- Resultados ---\n")
        f.write(f"Accuracy:  {accuracy:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1-Score:  {f1:.3f}\n")
        if auc_roc:
            f.write(f"AUC-ROC:   {auc_roc:.3f}\n")
        f.write(f"\nCV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n\n")
        
        f.write("--- Matriz de Confusión ---\n")
        f.write(f"                Predicho Seguro  Predicho Vulnerable\n")
        f.write(f"Real Seguro          {cm[0][0]:6d}            {cm[0][1]:6d}\n")
        f.write(f"Real Vulnerable      {cm[1][0]:6d}            {cm[1][1]:6d}\n")
    
    print(f"✅ Reporte guardado en: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return model, vectorizer, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrena el modelo con métricas completas")
    parser.add_argument("--dataset", default="data/msr_pipeline.csv",
                       help="Ruta al dataset procesado")
    parser.add_argument("--model", default="models/security_classifier_msr.joblib",
                       help="Ruta de salida del modelo")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proporción del dataset para test (default: 0.2)")
    parser.add_argument("--no-plots", action="store_true",
                       help="No generar gráficos")
    
    args = parser.parse_args()
    
    try:
        train_model_complete(
            args.dataset, 
            args.model, 
            args.test_size,
            not args.no_plots
        )
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
