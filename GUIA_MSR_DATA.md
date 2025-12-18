# 📘 Guía Completa: Trabajar con MSR_data_cleaned.csv

## 📋 Índice

1. [Descripción del Dataset](#descripción-del-dataset)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación Rápida](#instalación-rápida)
4. [Pipeline Completo Automatizado](#pipeline-completo-automatizado)
5. [Uso Paso a Paso](#uso-paso-a-paso)
6. [Scripts Disponibles](#scripts-disponibles)
7. [Estructura de Datos](#estructura-de-datos)
8. [Resultados Esperados](#resultados-esperados)
9. [Solución de Problemas](#solución-de-problemas)

---

## 📊 Descripción del Dataset

**MSR_data_cleaned.csv** es un dataset de vulnerabilidades de seguridad en código C/C++ que contiene:

- **~188,636 registros** de funciones vulnerables y sus parches
- **Columnas principales:**
  - `func_before`: Función vulnerable antes del parche
  - `func_after`: Función segura después del parche
  - `vul`: Flag de vulnerabilidad (1=vulnerable, 0=seguro)
  - `lang`: Lenguaje de programación (C, C++)
  - `commit_id`: Identificador del commit
  - `CVE ID`, `CWE ID`: Identificadores de vulnerabilidades
  - Metadatos adicionales (proyectos, parches, etc.)

---

## ⚙️ Requisitos Previos

### Software Necesario
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias Python
```txt
pandas>=2.1.0
numpy>=1.25.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🚀 Instalación Rápida

### 1. Clonar o descargar el proyecto

```powershell
cd "C:\Users\patri\OneDrive\Escritorio\Proyecto_UII_DSS"
```

### 2. Crear entorno virtual (recomendado)

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Si hay error de permisos, ejecutar:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Instalar dependencias

```powershell
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete en modo editable
pip install -e .
```

### 4. Verificar que MSR_data_cleaned.csv está presente

```powershell
# Debe estar en: data\MSR_data_cleaned.csv
dir data\MSR_data_cleaned.csv
```

---

## 🎯 Pipeline Completo Automatizado

### Opción 1: Ejecutar Pipeline Completo (Recomendado)

Este script ejecuta automáticamente todos los pasos del pipeline:

```powershell
python scripts\run_pipeline_completo.py
```

**Este comando ejecuta:**
1. ✅ Procesamiento de MSR_data_cleaned.csv → msr_pipeline.csv
2. ✅ Validación del dataset procesado
3. ✅ Entrenamiento del modelo con métricas completas
4. ✅ Pruebas de inferencia con ejemplos

**Duración estimada:** 30-60 minutos (depende del hardware)

---

## 📝 Uso Paso a Paso

Si prefieres ejecutar cada paso manualmente:

### Paso 1: Procesar MSR_data_cleaned.csv

Convierte el formato MSR al formato del pipeline (id, label, language, code):

```powershell
python scripts\proceso_msr.py
```

**Resultado:** `data/msr_pipeline.csv` (~377,000 registros)
- Se crean 2 filas por cada registro original:
  - `func_before` → etiqueta según `vul`
  - `func_after` → siempre "seguro"

**Opciones:**
```powershell
# Especificar archivos de entrada/salida
python scripts\proceso_msr.py --input data\MSR_data_cleaned.csv --output data\custom_output.csv
```

### Paso 2: Validar Dataset

Valida integridad, detecta nulos y analiza distribución de clases:

```powershell
python scripts\validar_datos.py data\msr_pipeline.csv --report
```

**Resultado:** `data/validation_report.txt`

**Validaciones realizadas:**
- ✅ Columnas requeridas presentes
- ✅ Sin valores nulos
- ✅ Etiquetas válidas
- ✅ Análisis de desbalance de clases
- ✅ Estadísticas de longitud de código

### Paso 3: Entrenar Modelo

Entrena un modelo RandomForest con validación cruzada:

```powershell
python scripts\entrenar_modelo.py
```

**Resultado:**
- Modelo: `models/security_classifier_msr.joblib`
- Reporte: `models/training_report.txt`
- Gráficos: `models/plots/confusion_matrix.png`, `feature_importance.png`

**Opciones:**
```powershell
# Personalizar entrenamiento
python scripts\entrenar_modelo.py --dataset data\msr_pipeline.csv --model models\mi_modelo.joblib --test-size 0.3

# Sin generar gráficos (más rápido)
python scripts\entrenar_modelo.py --no-plots
```

**Configuración del modelo:**
- 600 árboles (n_estimators=600)
- Profundidad máxima: 20
- class_weight='balanced_subsample' (maneja desbalance)
- Validación cruzada 5-fold

### Paso 4: Inferencia y Pruebas

#### 4.1 Probar con ejemplos predefinidos

```powershell
python scripts\inferencia_pruebas.py --test-examples
```

#### 4.2 Analizar un archivo específico

```powershell
python scripts\inferencia_pruebas.py --file demo_unsafe.c
python scripts\inferencia_pruebas.py --file demo_safe.c
```

#### 4.3 Modo interactivo

```powershell
python scripts\inferencia_pruebas.py --interactive
```

Luego ingresa código y finaliza con `END`.

#### 4.4 Análisis en batch

```powershell
# Analizar muestra del dataset
python scripts\inferencia_pruebas.py --dataset data\msr_pipeline.csv --sample 1000

# Analizar dataset completo (toma tiempo)
python scripts\inferencia_pruebas.py --dataset data\msr_pipeline.csv
```

**Resultado:** `data/predictions.csv`

---

## 📂 Scripts Disponibles

| Script | Descripción | Uso |
|--------|-------------|-----|
| `proceso_msr.py` | Convierte MSR_data_cleaned.csv al formato del pipeline | `python scripts\proceso_msr.py` |
| `validar_datos.py` | Valida integridad y calidad del dataset | `python scripts\validar_datos.py <dataset>` |
| `entrenar_modelo.py` | Entrena el modelo con métricas completas | `python scripts\entrenar_modelo.py` |
| `inferencia_pruebas.py` | Inferencia en archivos, batch o interactivo | `python scripts\inferencia_pruebas.py --help` |
| `run_pipeline_completo.py` | Ejecuta todo el pipeline automáticamente | `python scripts\run_pipeline_completo.py` |
| `analyze_msr_csv.py` | Analiza estructura del CSV original | `python scripts\analyze_msr_csv.py` |

---

## 🗂️ Estructura de Datos

### Formato Original (MSR_data_cleaned.csv)

```csv
commit_id, vul, lang, func_before, func_after, CVE ID, CWE ID, ...
abc123, 1, C, "void unsafe()...", "void safe()...", CVE-2020-1234, CWE-119, ...
```

### Formato del Pipeline (msr_pipeline.csv)

```csv
id, label, language, code
abc123_0_before, vulnerable, C, "void unsafe()..."
abc123_0_after, seguro, C, "void safe()..."
```

**Columnas requeridas:**
- `id`: Identificador único
- `label`: "vulnerable" o "seguro"
- `language`: Lenguaje de programación
- `code`: Código fuente

---

## 📈 Resultados Esperados

### Métricas del Modelo

Con el dataset completo (~377k muestras), se esperan métricas similares a:

```
Accuracy:   0.85-0.90
Precision:  0.75-0.85
Recall:     0.70-0.80
F1-Score:   0.70-0.80
```

**Nota:** El dataset está desbalanceado (más código seguro que vulnerable), por lo que el modelo usa `class_weight='balanced_subsample'`.

### Archivos Generados

```
data/
  msr_pipeline.csv          # Dataset procesado (~377k filas)
  validation_report.txt     # Reporte de validación
  predictions.csv           # Predicciones (si se ejecuta batch)

models/
  security_classifier_msr.joblib  # Modelo entrenado
  training_report.txt            # Métricas de entrenamiento
  plots/
    confusion_matrix.png         # Matriz de confusión
    feature_importance.png       # Importancia de features
```

---

## 🔧 Solución de Problemas

### Error: "MSR_data_cleaned.csv not found"

**Solución:** Verifica que el archivo esté en `data/MSR_data_cleaned.csv`

```powershell
dir data\MSR_data_cleaned.csv
```

### Error: "No module named 'pandas'"

**Solución:** Instala las dependencias

```powershell
pip install -r requirements.txt
```

### Error: "MemoryError" al procesar

**Solución:** El dataset es grande. Opciones:
1. Cerrar otras aplicaciones
2. Procesar por lotes (modificar scripts)
3. Aumentar memoria virtual de Windows

### Advertencia: "DtypeWarning: Columns have mixed types"

**Solución:** Esto es normal. El script maneja automáticamente los tipos mixtos con `low_memory=False`.

### Entrenamiento muy lento

**Solución:**
- Usa `--no-plots` para omitir generación de gráficos
- Reduce `n_estimators` en el código (línea del RandomForestClassifier)
- Usa una muestra del dataset para pruebas rápidas

---

## 🎓 Ejemplos de Uso

### Ejemplo 1: Pipeline completo desde cero

```powershell
# Activar entorno
.\.venv\Scripts\Activate.ps1

# Ejecutar pipeline completo
python scripts\run_pipeline_completo.py
```

### Ejemplo 2: Solo entrenar con datos procesados existentes

```powershell
# Si ya tienes msr_pipeline.csv
python scripts\entrenar_modelo.py --dataset data\msr_pipeline.csv
```

### Ejemplo 3: Analizar código vulnerable vs seguro

```powershell
# Analizar archivo unsafe
python scripts\inferencia_pruebas.py --file demo_unsafe.c

# Analizar archivo safe
python scripts\inferencia_pruebas.py --file demo_safe.c
```

### Ejemplo 4: Validar dataset antes de entrenar

```powershell
python scripts\validar_datos.py data\msr_pipeline.csv --report
```

---

## 🤝 Integración con CI/CD

Una vez entrenado el modelo, puedes integrarlo en tu pipeline CI/CD:

```yaml
# Ejemplo GitHub Actions
- name: Check Code Security
  run: python scripts/inferencia_pruebas.py --file ${{ matrix.file }}
```

Ver [README.md](README.md) principal para más detalles de integración CI/CD.

---

## 📚 Referencias

- **Dataset Original:** MSR (Mining Software Repositories)
- **BigVul Dataset:** Base de vulnerabilidades en C/C++
- **Proyecto:** Pipeline CI/CD con clasificador de vulnerabilidades

---

## ✅ Checklist de Configuración

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Paquete instalado (`pip install -e .`)
- [ ] MSR_data_cleaned.csv en `data/`
- [ ] Pipeline ejecutado con éxito
- [ ] Modelo entrenado en `models/`
- [ ] Pruebas de inferencia funcionando

---

## 📧 Soporte

Para problemas o preguntas:
1. Revisa la sección [Solución de Problemas](#solución-de-problemas)
2. Verifica los logs en `logs/`
3. Consulta los reportes generados en `models/` y `data/`

---

**Última actualización:** Diciembre 2025  
**Versión:** 1.0
