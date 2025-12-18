"""
Script para procesar MSR_data_cleaned.csv y convertirlo al formato del pipeline.

El formato esperado por el pipeline es:
- id: identificador único
- label: "vulnerable" o "seguro"
- language: lenguaje de programación
- code: código fuente

El dataset MSR tiene:
- func_before: función vulnerable antes del parche
- func_after: función segura después del parche
- vul: flag de vulnerabilidad (1=vulnerable, 0=seguro)
- lang: lenguaje (C, C++, etc.)
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np


def process_msr_csv(input_path: str = "data/MSR_data_cleaned.csv", 
                    output_path: str = "data/msr_pipeline.csv"):
    """
    Procesa MSR_data_cleaned.csv y lo convierte al formato del pipeline.
    
    Crea dos filas por cada entrada original:
    - Una para func_before (vulnerable si vul==1)
    - Una para func_after (siempre segura)
    """
    
    print("🔄 Procesando MSR_data_cleaned.csv...")
    print("=" * 70)
    
    # Leer CSV
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Error: {input_path} no encontrado")
        sys.exit(1)
    
    print(f"📂 Leyendo {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"✓ Cargadas {len(df)} filas con {len(df.columns)} columnas")
    
    # Crear registros para el pipeline
    records = []
    processed = 0
    skipped_before = 0
    skipped_after = 0
    
    print(f"\n🔧 Procesando funciones...")
    
    for idx, row in df.iterrows():
        # Extraer información base
        commit_id = str(row.get('commit_id', '')) or f"row{idx}"
        vul_flag = row.get('vul')
        lang = str(row.get('lang', 'C')).strip()
        
        # Determinar si es vulnerable
        is_vulnerable = False
        if pd.notna(vul_flag):
            if isinstance(vul_flag, (int, float)):
                is_vulnerable = (vul_flag == 1)
            elif isinstance(vul_flag, str):
                is_vulnerable = vul_flag.lower() in ['1', 'true', 'yes']
        
        func_before = row.get('func_before')
        func_after = row.get('func_after')
        
        # Procesar func_before (vulnerable si vul==1)
        if pd.notna(func_before) and str(func_before).strip():
            code_before = str(func_before).strip()
            if len(code_before) > 20:  # Validación mínima
                records.append({
                    'id': f"{commit_id}_{idx}_before",
                    'label': 'vulnerable' if is_vulnerable else 'seguro',
                    'language': lang,
                    'code': code_before
                })
            else:
                skipped_before += 1
        else:
            skipped_before += 1
        
        # Procesar func_after (siempre segura)
        if pd.notna(func_after) and str(func_after).strip():
            code_after = str(func_after).strip()
            if len(code_after) > 20:  # Validación mínima
                records.append({
                    'id': f"{commit_id}_{idx}_after",
                    'label': 'seguro',
                    'language': lang,
                    'code': code_after
                })
            else:
                skipped_after += 1
        else:
            skipped_after += 1
        
        processed += 1
        if processed % 10000 == 0:
            print(f"  Procesadas {processed}/{len(df)} filas...")
    
    # Crear DataFrame de salida
    output_df = pd.DataFrame(records)
    
    print(f"\n📊 Estadísticas de procesamiento:")
    print(f"  - Filas originales: {len(df)}")
    print(f"  - Registros generados: {len(output_df)}")
    print(f"  - func_before omitidas: {skipped_before}")
    print(f"  - func_after omitidas: {skipped_after}")
    
    print(f"\n📈 Distribución de etiquetas:")
    label_counts = output_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(output_df)) * 100
        print(f"  - {label}: {count} ({percentage:.2f}%)")
    
    print(f"\n🗣️ Lenguajes detectados:")
    lang_counts = output_df['language'].value_counts()
    for lang, count in lang_counts.items():
        print(f"  - {lang}: {count}")
    
    # Guardar resultado
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Archivo guardado en: {output_path}")
    print(f"📦 Tamaño del archivo: {output_file.stat().st_size / 1024**2:.2f} MB")
    print("=" * 70)
    
    return output_df


def validate_output(df: pd.DataFrame):
    """Valida que el DataFrame de salida tenga el formato correcto."""
    
    print("\n🔍 Validando formato de salida...")
    
    required_columns = {'id', 'label', 'language', 'code'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        print(f"❌ Faltan columnas requeridas: {missing_columns}")
        return False
    
    # Validar que no haya valores nulos en columnas críticas
    nulls = df[list(required_columns)].isnull().sum()
    if nulls.any():
        print("⚠️  Valores nulos detectados:")
        for col, count in nulls[nulls > 0].items():
            print(f"  - {col}: {count}")
        return False
    
    # Validar etiquetas
    valid_labels = {'vulnerable', 'seguro'}
    invalid_labels = set(df['label'].unique()) - valid_labels
    if invalid_labels:
        print(f"⚠️  Etiquetas inválidas detectadas: {invalid_labels}")
        return False
    
    print("✅ Formato de salida válido")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Procesa MSR_data_cleaned.csv al formato del pipeline")
    parser.add_argument("--input", default="data/MSR_data_cleaned.csv", 
                       help="Ruta del archivo MSR de entrada")
    parser.add_argument("--output", default="data/msr_pipeline.csv",
                       help="Ruta del archivo procesado de salida")
    
    args = parser.parse_args()
    
    # Procesar
    result_df = process_msr_csv(args.input, args.output)
    
    # Validar
    if validate_output(result_df):
        print("\n🎉 Procesamiento completado exitosamente")
        print(f"\n💡 Ahora puedes entrenar el modelo con:")
        print(f"   python -m secure_pipeline.train --dataset {args.output}")
    else:
        print("\n⚠️  Procesamiento completado con advertencias")
        sys.exit(1)
