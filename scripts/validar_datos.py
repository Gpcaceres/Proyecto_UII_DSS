"""
Script de validación integral para datos del pipeline de seguridad.

Valida:
- Integridad estructural (columnas requeridas)
- Valores nulos y duplicados
- Distribución de clases (desbalance)
- Calidad del código fuente
- Longitud y características del código
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
from collections import Counter


def validate_dataset(dataset_path: str):
    """Validación completa del dataset."""
    
    print("🔍 Validación del Dataset")
    print("=" * 70)
    
    # 1. Verificar existencia
    path = Path(dataset_path)
    if not path.exists():
        print(f"❌ Error: {dataset_path} no encontrado")
        return False
    
    print(f"📂 Archivo: {dataset_path}")
    print(f"📦 Tamaño: {path.stat().st_size / 1024**2:.2f} MB\n")
    
    # 2. Cargar dataset
    try:
        df = pd.read_csv(dataset_path, low_memory=False)
        print(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas\n")
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return False
    
    all_valid = True
    
    # 3. Validar estructura
    print("📋 Validación de Estructura")
    print("-" * 70)
    
    required_columns = {'id', 'label', 'language', 'code'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        print(f"❌ Faltan columnas requeridas: {missing_columns}")
        all_valid = False
    else:
        print(f"✅ Todas las columnas requeridas presentes: {required_columns}")
    
    print(f"\nColumnas actuales: {list(df.columns)}\n")
    
    # 4. Validar valores nulos
    print("🔎 Validación de Valores Nulos")
    print("-" * 70)
    
    null_counts = df.isnull().sum()
    has_nulls = False
    
    for col in required_columns:
        if col in df.columns:
            null_count = null_counts[col]
            null_pct = (null_count / len(df)) * 100
            
            if null_count > 0:
                print(f"⚠️  {col}: {null_count} nulos ({null_pct:.2f}%)")
                has_nulls = True
                all_valid = False
            else:
                print(f"✅ {col}: Sin valores nulos")
    
    if not has_nulls:
        print("✅ No hay valores nulos en columnas críticas")
    
    print()
    
    # 5. Validar etiquetas
    print("🏷️  Validación de Etiquetas")
    print("-" * 70)
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("Distribución de clases:")
        
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {label}: {count} ({percentage:.2f}%)")
        
        # Detectar desbalance
        if len(label_counts) >= 2:
            max_class = label_counts.max()
            min_class = label_counts.min()
            imbalance_ratio = max_class / min_class
            
            print(f"\n📊 Ratio de desbalance: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print("⚠️  ADVERTENCIA: Desbalance severo de clases")
                print("   Recomendación: Usar class_weight='balanced' en el modelo")
            elif imbalance_ratio > 3:
                print("⚠️  Desbalance moderado de clases")
        
        # Validar valores válidos
        valid_labels = {'vulnerable', 'seguro'}
        invalid_labels = set(df['label'].unique()) - valid_labels
        
        if invalid_labels:
            print(f"\n❌ Etiquetas inválidas detectadas: {invalid_labels}")
            all_valid = False
        else:
            print("\n✅ Todas las etiquetas son válidas")
    
    print()
    
    # 6. Validar lenguajes
    print("🗣️  Validación de Lenguajes")
    print("-" * 70)
    
    if 'language' in df.columns:
        lang_counts = df['language'].value_counts()
        print("Lenguajes detectados:")
        
        for lang, count in lang_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {lang}: {count} ({percentage:.2f}%)")
        
        print(f"\n✅ Total de lenguajes: {len(lang_counts)}")
    
    print()
    
    # 7. Validar código
    print("💻 Validación de Código Fuente")
    print("-" * 70)
    
    if 'code' in df.columns:
        # Longitud del código
        code_lengths = df['code'].astype(str).str.len()
        
        print("Estadísticas de longitud:")
        print(f"  - Mínimo: {code_lengths.min()} caracteres")
        print(f"  - Máximo: {code_lengths.max()} caracteres")
        print(f"  - Promedio: {code_lengths.mean():.0f} caracteres")
        print(f"  - Mediana: {code_lengths.median():.0f} caracteres")
        
        # Detectar código muy corto (posiblemente inválido)
        too_short = (code_lengths < 20).sum()
        if too_short > 0:
            print(f"\n⚠️  {too_short} registros con código muy corto (<20 chars)")
        
        # Detectar código muy largo (posiblemente problemático)
        too_long = (code_lengths > 10000).sum()
        if too_long > 0:
            print(f"⚠️  {too_long} registros con código muy largo (>10k chars)")
        
        # Líneas de código
        df['num_lines'] = df['code'].astype(str).str.count('\n') + 1
        print(f"\nLíneas de código:")
        print(f"  - Promedio: {df['num_lines'].mean():.1f} líneas")
        print(f"  - Mediana: {df['num_lines'].median():.1f} líneas")
        
        print("\n✅ Validación de código completada")
    
    print()
    
    # 8. Validar duplicados
    print("🔄 Validación de Duplicados")
    print("-" * 70)
    
    # Duplicados por ID
    if 'id' in df.columns:
        duplicated_ids = df['id'].duplicated().sum()
        if duplicated_ids > 0:
            print(f"⚠️  {duplicated_ids} IDs duplicados")
            all_valid = False
        else:
            print("✅ No hay IDs duplicados")
    
    # Duplicados de código
    if 'code' in df.columns:
        duplicated_code = df['code'].duplicated().sum()
        if duplicated_code > 0:
            dup_pct = (duplicated_code / len(df)) * 100
            print(f"⚠️  {duplicated_code} códigos duplicados ({dup_pct:.2f}%)")
        else:
            print("✅ No hay códigos duplicados")
    
    print()
    
    # 9. Resumen final
    print("=" * 70)
    if all_valid:
        print("✅ DATASET VÁLIDO - Listo para entrenamiento")
        return True
    else:
        print("⚠️  DATASET CON ADVERTENCIAS - Revisar problemas detectados")
        return False


def generate_statistics_report(dataset_path: str, output_path: str = "data/validation_report.txt"):
    """Genera un reporte detallado de estadísticas."""
    
    df = pd.read_csv(dataset_path, low_memory=False)
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("REPORTE DE VALIDACIÓN DEL DATASET")
    report_lines.append("=" * 70)
    report_lines.append(f"\nArchivo: {dataset_path}")
    report_lines.append(f"Fecha: {pd.Timestamp.now()}")
    report_lines.append(f"\nTotal de registros: {len(df)}")
    report_lines.append(f"Columnas: {list(df.columns)}")
    
    if 'label' in df.columns:
        report_lines.append("\n--- Distribución de Clases ---")
        for label, count in df['label'].value_counts().items():
            pct = (count / len(df)) * 100
            report_lines.append(f"{label}: {count} ({pct:.2f}%)")
    
    if 'language' in df.columns:
        report_lines.append("\n--- Distribución de Lenguajes ---")
        for lang, count in df['language'].value_counts().items():
            pct = (count / len(df)) * 100
            report_lines.append(f"{lang}: {count} ({pct:.2f}%)")
    
    report = "\n".join(report_lines)
    
    # Guardar reporte
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding='utf-8')
    
    print(f"📄 Reporte guardado en: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Valida el dataset del pipeline")
    parser.add_argument("dataset", help="Ruta al archivo CSV del dataset")
    parser.add_argument("--report", action="store_true", 
                       help="Generar reporte detallado en archivo")
    
    args = parser.parse_args()
    
    # Validar
    is_valid = validate_dataset(args.dataset)
    
    # Generar reporte si se solicita
    if args.report:
        generate_statistics_report(args.dataset)
    
    # Exit code
    sys.exit(0 if is_valid else 1)
