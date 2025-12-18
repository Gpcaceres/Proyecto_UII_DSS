# Pipeline CI/CD seguro con clasificador clásico

Este repositorio provee el esqueleto mínimo para entrenar y usar un modelo **no LLM** que clasifica código fuente como `seguro` o `vulnerable` dentro de un pipeline CI/CD con enfoque *shift-left*. Incluye:

- Dataset de demostración (`data/demo_dataset.csv`) con fragmentos Python etiquetados.
- **Dataset MSR_data_cleaned.csv** con ~188k vulnerabilidades reales en C/C++.
- Extracción de features clásicas (tokens, profundidad AST, llamadas peligrosas y sanitización).
- Entrenamiento de un modelo de **RandomForest** con validación cruzada y guardado en `.joblib`.
- Inferencia en archivos de código para integrarlo en jobs de revisión de seguridad.
- **Scripts completos** para procesamiento, validación, entrenamiento e inferencia.

---

## 🚀 Inicio Rápido con MSR_data_cleaned.csv

**¡NUEVO!** Pipeline completo automatizado para trabajar con MSR_data_cleaned.csv:

```powershell
# 1. Instalar dependencias
pip install -r requirements.txt
pip install -e .

# 2. Ejecutar pipeline completo (procesar, validar, entrenar, probar)
python scripts\run_pipeline_completo.py
```

**📘 Ver [GUIA_MSR_DATA.md](GUIA_MSR_DATA.md) para documentación completa y detallada.**

### Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `scripts/run_pipeline_completo.py` | **Pipeline completo automatizado** |
| `scripts/proceso_msr.py` | Procesa MSR_data_cleaned.csv → msr_pipeline.csv |
| `scripts/validar_datos.py` | Valida integridad del dataset |
| `scripts/entrenar_modelo.py` | Entrena modelo con métricas completas |
| `scripts/inferencia_pruebas.py` | Inferencia en archivos, batch o modo interactivo |

---

## Dataset BigVul / MSR (C/C++)

### Opción 1: Usar el pipeline automatizado (Recomendado)

```powershell
python scripts\run_pipeline_completo.py
```

### Opción 2: Usar módulos del paquete

El proyecto incluye conversores para formato MSR:

```bash
# Convertir desde JSON
python -m secure_pipeline.convert_bigvul --input data/MSR_data_cleaned.json --output data/bigvul_pipeline.csv

# Entrenar directamente
python -m secure_pipeline.train --dataset data/MSR_data_cleaned.json --model-path models/security_classifier.joblib
```

### Opción 3: Scripts paso a paso

```powershell
# 1. Procesar
python scripts\proceso_msr.py

# 2. Validar
python scripts\validar_datos.py data\msr_pipeline.csv

# 3. Entrenar
python scripts\entrenar_modelo.py

# 4. Probar
python scripts\inferencia_pruebas.py --test-examples
```

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

