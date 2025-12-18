"""
Script de inferencia y pruebas del modelo entrenado.

Permite:
1. Cargar el modelo entrenado
2. Analizar archivos de código individuales
3. Ejecutar pruebas en batch sobre múltiples archivos
4. Mostrar explicaciones de las predicciones
"""

import joblib
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


# Importar desde el módulo del pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from secure_pipeline.features import extract_features


def load_model(model_path: str = "models/security_classifier_balanced.joblib"):
    """Carga el modelo entrenado (por defecto el modelo balanceado en producción)."""
    
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    print(f"📦 Cargando modelo desde: {model_path}")
    model_data = joblib.load(path)
    
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    metrics = model_data.get('metrics', {})
    
    print(f"✅ Modelo cargado")
    
    if metrics:
        print(f"\n📊 Métricas del modelo:")
        print(f"  - Accuracy:  {metrics.get('accuracy', 'N/A'):.3f}")
        print(f"  - Precision: {metrics.get('precision', 'N/A'):.3f}")
        print(f"  - Recall:    {metrics.get('recall', 'N/A'):.3f}")
        print(f"  - F1-Score:  {metrics.get('f1', 'N/A'):.3f}")
    
    return model, vectorizer, model_data


def predict_code(code: str, model, vectorizer, language: str = "C", verbose: bool = True):
    """Predice si un código es vulnerable."""
    
    # Extraer features
    features = extract_features(code, language)
    features_vec = vectorizer.transform([features])
    
    # Predicción
    prediction = model.predict(features_vec)[0]
    probability = model.predict_proba(features_vec)[0]
    
    label = "vulnerable" if prediction == 1 else "seguro"
    confidence = probability[prediction]
    
    if verbose:
        print(f"\n🔍 Resultado de análisis:")
        print(f"  - Clasificación: {label.upper()}")
        print(f"  - Confianza: {confidence*100:.1f}%")
        print(f"  - Probabilidad vulnerable: {probability[1]*100:.1f}%")
        print(f"  - Probabilidad seguro: {probability[0]*100:.1f}%")
        
        # Mostrar features extraídas
        print(f"\n📋 Features detectadas:")
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)) and value != 0:
                print(f"  - {key}: {value}")
    
    return {
        'label': label,
        'prediction': int(prediction),
        'confidence': float(confidence),
        'probability_vulnerable': float(probability[1]),
        'probability_safe': float(probability[0]),
        'features': features
    }


def analyze_file(file_path: str, model, vectorizer, language: str = None):
    """Analiza un archivo de código completo."""
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: Archivo no encontrado: {file_path}")
        return None
    
    # Detectar lenguaje por extensión si no se proporciona
    if language is None:
        ext = path.suffix.lower()
        language_map = {
            '.c': 'C',
            '.cpp': 'C++',
            '.cc': 'C++',
            '.h': 'C',
            '.hpp': 'C++',
            '.py': 'Python',
            '.java': 'Java',
            '.js': 'JavaScript'
        }
        language = language_map.get(ext, 'C')
    
    print(f"\n{'='*70}")
    print(f"📄 Analizando archivo: {file_path}")
    print(f"🗣️  Lenguaje: {language}")
    print(f"{'='*70}")
    
    # Leer código
    try:
        code = path.read_text(encoding='utf-8')
    except:
        code = path.read_text(encoding='latin-1')
    
    print(f"📏 Tamaño: {len(code)} caracteres, {code.count(chr(10))+1} líneas")
    
    # Predecir
    result = predict_code(code, model, vectorizer, language)
    
    return result


def batch_analyze(dataset_path: str, model, vectorizer, 
                 output_path: str = "data/predictions.csv",
                 sample_size: int = None):
    """Analiza múltiples muestras del dataset."""
    
    print(f"\n🔄 Análisis en batch")
    print(f"{'='*70}")
    
    # Cargar dataset
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"📂 Dataset: {dataset_path}")
    print(f"📊 Total de muestras: {len(df)}")
    
    # Tomar muestra si se especifica
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"📌 Analizando muestra de {sample_size} registros")
    
    # Preparar resultados
    results = []
    correct = 0
    total = 0
    
    print(f"\n🔍 Analizando muestras...")
    
    for idx, row in df.iterrows():
        code = row.get('code', '')
        true_label = row.get('label', 'unknown')
        language = row.get('language', 'C')
        
        # Predecir
        result = predict_code(code, model, vectorizer, language, verbose=False)
        
        # Comparar con etiqueta real si existe
        if true_label in ['vulnerable', 'seguro']:
            is_correct = (result['label'] == true_label)
            if is_correct:
                correct += 1
            total += 1
        else:
            is_correct = None
        
        # Guardar resultado
        results.append({
            'id': row.get('id', idx),
            'true_label': true_label,
            'predicted_label': result['label'],
            'confidence': result['confidence'],
            'probability_vulnerable': result['probability_vulnerable'],
            'correct': is_correct,
            'language': language
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Procesadas {idx + 1} muestras...")
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    # Guardar
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    
    print(f"\n✅ Resultados guardados en: {output_path}")
    
    # Mostrar estadísticas
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n📈 Estadísticas:")
        print(f"  - Total analizadas: {total}")
        print(f"  - Correctas: {correct}")
        print(f"  - Accuracy: {accuracy:.2f}%")
        
        # Por clase
        for label in ['vulnerable', 'seguro']:
            subset = results_df[results_df['true_label'] == label]
            if len(subset) > 0:
                correct_class = subset['correct'].sum()
                total_class = len(subset)
                acc_class = (correct_class / total_class) * 100
                print(f"  - Accuracy {label}: {acc_class:.2f}% ({correct_class}/{total_class})")
    
    return results_df


def interactive_mode(model, vectorizer):
    """Modo interactivo para análisis de código."""
    
    print(f"\n{'='*70}")
    print("🎮 MODO INTERACTIVO")
    print("{'='*70}")
    print("Ingresa código para analizar (finaliza con una línea que contenga solo 'END')")
    print("O escribe 'exit' para salir\n")
    
    while True:
        print("\n" + "-"*70)
        print("Ingresa el código:")
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            if line.strip().lower() == 'exit':
                print("\n👋 Saliendo...")
                return
            lines.append(line)
        
        if not lines:
            continue
        
        code = '\n'.join(lines)
        
        # Solicitar lenguaje
        language = input("\nLenguaje (C/C++/Python/Java) [C]: ").strip() or "C"
        
        # Analizar
        result = predict_code(code, model, vectorizer, language)
        
        # Preguntar si continuar
        cont = input("\n¿Analizar otro código? (s/n) [s]: ").strip().lower()
        if cont == 'n':
            print("\n👋 Saliendo...")
            break


def test_with_examples(model, vectorizer):
    """Prueba el modelo con ejemplos predefinidos."""
    
    print(f"\n{'='*70}")
    print("🧪 PRUEBAS CON EJEMPLOS PREDEFINIDOS")
    print(f"{'='*70}")
    
    examples = [
        {
            'name': 'Buffer overflow vulnerable',
            'code': '''
void unsafe_copy(char *dest, char *src) {
    strcpy(dest, src);  // Sin verificación de tamaño
}
''',
            'language': 'C',
            'expected': 'vulnerable'
        },
        {
            'name': 'Buffer overflow seguro',
            'code': '''
void safe_copy(char *dest, char *src, size_t size) {
    strncpy(dest, src, size - 1);
    dest[size - 1] = '\\0';
}
''',
            'language': 'C',
            'expected': 'seguro'
        },
        {
            'name': 'SQL Injection vulnerable',
            'code': '''
def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return db.execute(query)
''',
            'language': 'Python',
            'expected': 'vulnerable'
        },
        {
            'name': 'SQL Injection seguro',
            'code': '''
def get_user(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))
''',
            'language': 'Python',
            'expected': 'seguro'
        }
    ]
    
    correct = 0
    total = len(examples)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Ejemplo {i}: {example['name']}")
        print(f"   Esperado: {example['expected']}")
        
        result = predict_code(example['code'], model, vectorizer, 
                            example['language'], verbose=False)
        
        is_correct = (result['label'] == example['expected'])
        status = "✅" if is_correct else "❌"
        
        print(f"   Predicho: {result['label']} (confianza: {result['confidence']*100:.1f}%) {status}")
        
        if is_correct:
            correct += 1
    
    print(f"\n{'='*70}")
    print(f"📊 Resultados: {correct}/{total} correctos ({correct/total*100:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencia con el modelo entrenado")
    parser.add_argument("--model", default="models/security_classifier_balanced.joblib",
                       help="Ruta al modelo entrenado (por defecto: modelo balanceado en producción)")
    parser.add_argument("--file", help="Archivo de código a analizar")
    parser.add_argument("--dataset", help="Dataset para análisis en batch")
    parser.add_argument("--sample", type=int, help="Tamaño de muestra para batch")
    parser.add_argument("--interactive", action="store_true",
                       help="Modo interactivo")
    parser.add_argument("--test-examples", action="store_true",
                       help="Probar con ejemplos predefinidos")
    parser.add_argument("--language", help="Lenguaje del código")
    
    args = parser.parse_args()
    
    try:
        # Cargar modelo
        model, vectorizer, model_data = load_model(args.model)
        
        # Ejecutar según modo
        if args.test_examples:
            test_with_examples(model, vectorizer)
        
        elif args.file:
            result = analyze_file(args.file, model, vectorizer, args.language)
        
        elif args.dataset:
            batch_analyze(args.dataset, model, vectorizer, sample_size=args.sample)
        
        elif args.interactive:
            interactive_mode(model, vectorizer)
        
        else:
            print("\n⚠️  Debes especificar una opción:")
            print("  --file <archivo>         Analizar un archivo")
            print("  --dataset <dataset>      Análisis en batch")
            print("  --interactive            Modo interactivo")
            print("  --test-examples          Probar con ejemplos")
            print("\nUsa --help para más información")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
