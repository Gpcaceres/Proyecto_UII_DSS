# Pipeline CI/CD seguro con clasificador clásico

Este repositorio provee el esqueleto mínimo para entrenar y usar un modelo **no LLM** que clasifica código fuente como `seguro` o `vulnerable` dentro de un pipeline CI/CD con enfoque *shift-left*. Incluye:

- Dataset de demostración (`data/demo_dataset.csv`) con fragmentos Python etiquetados.
- Extracción de features clásicas (tokens, profundidad AST, llamadas peligrosas y sanitización).
- Entrenamiento de un modelo de **RandomForest** con validación cruzada y guardado en `.joblib`.
- Inferencia en archivos de código para integrarlo en jobs de revisión de seguridad.

---

## Dataset BigVul (C/C++)

Este proyecto incluye un conversor:

```
python -m secure_pipeline.convert_bigvul --input data/demo_dataset.csv --output data/bigvul_pipeline.csv
```

El conversor transforma el dataset BigVul (`MSR_data_cleaned.json`) al formato estándar del pipeline:

```bash
# Si tienes el JSON limpio en data/, puedes convertirlo y entrenar así:
python -m secure_pipeline.convert_bigvul --input data/MSR_data_cleaned.json --output data/bigvul_pipeline.csv
python -m secure_pipeline.train --dataset data/MSR_data_cleaned.json --model-path models/security_classifier.joblib
```

El archivo resultante (`data/bigvul_pipeline.csv`) puede entrenar modelos para vulnerabilidades en C/C++.

---

# 🚀 Uso rápido en Windows (PowerShell)

```powershell
cd "C:\Users\patri\OneDrive\Escritorio\pipeline-ci-cd"

# Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Actualizar pip (opcional)
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo editable
pip install -e .

# Convertir dataset BigVul (si tienes JSON limpio)
python -m secure_pipeline.convert_bigvul --input data/MSR_data_cleaned.json --output data/bigvul_pipeline.csv

# Entrenar modelo con BigVul
python -m secure_pipeline.train --dataset data/bigvul_pipeline.csv --model-path models/security_classifier.joblib

# Entrenar directamente desde JSON
python -m secure_pipeline.train --dataset data/MSR_data_cleaned.json --model-path models/security_classifier.joblib

# Inferencia
python -m secure_pipeline.infer sample.py
```

---

## 📊 Resultados del modelo (BigVul)

Entrenamiento con 199k funciones C/C++:

- Accuracy validación cruzada: **0.852**
- Accuracy global: **0.89**
- Recall vulnerable: **0.75**
- F1 vulnerable: **0.42**

El modelo logra identificar vulnerabilidades reales en C/C++ con un desempeño robusto pese al desbalance extremo del dataset.

---

## 🔧 Integración con CI/CD

1. **Pull Request → Ejecuta el clasificador**
2. Los archivos modificados son evaluados por:
   ```
   python -m secure_pipeline.infer archivo.cpp
   ```
3. Si la predicción devuelve `vulnerable`, el pipeline **bloquea el merge**.
4. Notificación por Telegram, Slack o email con el JSON de predicción.
5. Si pasa, continúa a pruebas automatizadas y despliegue.

---

## 📁 Estructura del repositorio

```
secure_pipeline/
 ├── data.py
 ├── features.py
 ├── train.py
 ├── infer.py
 ├── convert_bigvul.py
data/
 ├── demo_dataset.csv
 ├── MSR_data_cleaned.json
 ├── bigvul_pipeline.csv
models/
 └── security_classifier.joblib
```

---

## 📜 Licencia

Proyecto educativo para investigación y prácticas de CI/CD seguro.  


---

