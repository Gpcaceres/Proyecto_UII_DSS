"""
Script maestro para ejecutar el pipeline completo de MSR_data_cleaned.csv

Este script automatiza todo el flujo:
1. Procesar MSR_data_cleaned.csv al formato del pipeline
2. Validar el dataset procesado
3. Entrenar el modelo
4. Ejecutar pruebas de inferencia
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_step(step_name: str, command: list, critical: bool = True):
    """Ejecuta un paso del pipeline."""
    
    print(f"\n{'='*70}")
    print(f"🚀 PASO: {step_name}")
    print(f"{'='*70}")
    print(f"Comando: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False, text=True)
        print(f"\n✅ {step_name} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en {step_name}")
        print(f"Código de salida: {e.returncode}")
        if critical:
            print("⚠️  Este es un paso crítico. Deteniendo el pipeline.")
            sys.exit(1)
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        if critical:
            sys.exit(1)
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║       PIPELINE COMPLETO: MSR_data_cleaned.csv                   ║
    ║       Procesamiento, Validación, Entrenamiento e Inferencia     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    inicio = datetime.now()
    print(f"⏰ Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Verificar que existe el archivo original
    if not Path("data/MSR_data_cleaned.csv").exists():
        print("❌ Error: data/MSR_data_cleaned.csv no encontrado")
        print("   Por favor, asegúrate de que el archivo esté en la carpeta data/")
        sys.exit(1)
    
    # Paso 1: Procesar MSR_data_cleaned.csv
    success_1 = run_step(
        "1/4 - Procesamiento de MSR_data_cleaned.csv",
        ["python", "scripts/proceso_msr.py"],
        critical=True
    )
    
    # Paso 2: Validar dataset procesado
    success_2 = run_step(
        "2/4 - Validación del dataset procesado",
        ["python", "scripts/validar_datos.py", "data/msr_pipeline.csv", "--report"],
        critical=True
    )
    
    # Paso 3: Entrenar modelo
    success_3 = run_step(
        "3/4 - Entrenamiento del modelo",
        ["python", "scripts/entrenar_modelo.py", 
         "--dataset", "data/msr_pipeline.csv",
         "--model", "models/security_classifier_msr.joblib"],
        critical=True
    )
    
    # Paso 4: Pruebas con ejemplos
    success_4 = run_step(
        "4/4 - Pruebas de inferencia",
        ["python", "scripts/inferencia_pruebas.py",
         "--model", "models/security_classifier_msr.joblib",
         "--test-examples"],
        critical=False
    )
    
    # Resumen final
    fin = datetime.now()
    duracion = fin - inicio
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETADO")
    print(f"{'='*70}")
    print(f"⏰ Inicio:    {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Fin:       {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duración:  {duracion}")
    print(f"\n📊 Resultados:")
    print(f"  ✓ Dataset procesado: data/msr_pipeline.csv")
    print(f"  ✓ Modelo entrenado: models/security_classifier_msr.joblib")
    print(f"  ✓ Reporte de entrenamiento: models/training_report.txt")
    print(f"  ✓ Reporte de validación: data/validation_report.txt")
    
    if Path("models/plots").exists():
        print(f"  ✓ Gráficos: models/plots/")
    
    print(f"\n🎯 Próximos pasos:")
    print(f"  1. Revisar métricas en models/training_report.txt")
    print(f"  2. Analizar archivos con:")
    print(f"     python scripts/inferencia_pruebas.py --file <archivo.c>")
    print(f"  3. Modo interactivo:")
    print(f"     python scripts/inferencia_pruebas.py --interactive")
    print(f"  4. Análisis batch:")
    print(f"     python scripts/inferencia_pruebas.py --dataset data/msr_pipeline.csv --sample 1000")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
