"""Script para analizar MSR_data_cleaned.csv y configurar el pipeline"""
import pandas as pd
import sys
from pathlib import Path

def analyze_msr_csv():
    csv_path = Path("data/MSR_data_cleaned.csv")
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} no encontrado")
        sys.exit(1)
    
    print("📊 Analizando MSR_data_cleaned.csv...")
    print("=" * 70)
    
    # Leer CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n📋 Información básica:")
    print(f"  - Filas: {len(df)}")
    print(f"  - Columnas: {len(df.columns)}")
    print(f"  - Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📝 Columnas detectadas:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} ({df[col].dtype})")
    
    print(f"\n🔍 Primeras 3 filas:")
    print(df.head(3))
    
    print(f"\n📊 Valores nulos por columna:")
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Detectar columnas clave
    print(f"\n🔑 Columnas clave detectadas:")
    key_columns = []
    if 'vul' in df.columns:
        key_columns.append('vul')
        print(f"  ✓ vul: {df['vul'].value_counts().to_dict()}")
    if 'vulnerable' in df.columns:
        key_columns.append('vulnerable')
        print(f"  ✓ vulnerable: {df['vulnerable'].value_counts().to_dict()}")
    if 'label' in df.columns:
        key_columns.append('label')
        print(f"  ✓ label: {df['label'].value_counts().to_dict()}")
    if 'lang' in df.columns:
        key_columns.append('lang')
        print(f"  ✓ lang: {df['lang'].value_counts().to_dict()}")
    if 'language' in df.columns:
        key_columns.append('language')
        print(f"  ✓ language: {df['language'].value_counts().to_dict()}")
    
    # Detectar columnas de código
    code_columns = []
    for col in ['func_before', 'func_after', 'code', 'content', 'snippet']:
        if col in df.columns:
            code_columns.append(col)
            sample = df[col].iloc[0] if not df[col].isnull().all() else "N/A"
            sample_preview = str(sample)[:100] + "..." if len(str(sample)) > 100 else str(sample)
            print(f"  ✓ {col}: presente (ejemplo: {sample_preview})")
    
    if not code_columns:
        print("  ⚠️  No se detectaron columnas de código estándar")
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    
    return df

if __name__ == "__main__":
    df = analyze_msr_csv()
