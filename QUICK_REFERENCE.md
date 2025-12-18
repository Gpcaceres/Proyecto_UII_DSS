# 🚀 Referencia Rápida: MSR_data_cleaned.csv

## ⚡ Comandos Esenciales

### Pipeline Completo (Recomendado)
```powershell
python scripts\run_pipeline_completo.py
```
✅ Procesa, valida, entrena y prueba automáticamente

---

## 📋 Comandos Individuales

### 1️⃣ Procesar Dataset
```powershell
python scripts\proceso_msr.py
```
**Entrada:** `data/MSR_data_cleaned.csv`  
**Salida:** `data/msr_pipeline.csv`

### 2️⃣ Validar Dataset
```powershell
python scripts\validar_datos.py data\msr_pipeline.csv --report
```
**Salida:** `data/validation_report.txt`

### 3️⃣ Entrenar Modelo
```powershell
python scripts\entrenar_modelo.py
```
**Salida:** 
- `models/security_classifier_msr.joblib`
- `models/training_report.txt`
- `models/plots/`

### 4️⃣ Probar Modelo

#### Ejemplos predefinidos
```powershell
python scripts\inferencia_pruebas.py --test-examples
```

#### Analizar archivo
```powershell
python scripts\inferencia_pruebas.py --file demo_unsafe.c
```

#### Modo interactivo
```powershell
python scripts\inferencia_pruebas.py --interactive
```

#### Análisis batch
```powershell
python scripts\inferencia_pruebas.py --dataset data\msr_pipeline.csv --sample 1000
```

---

## 📊 Estructura de Archivos

```
data/
  MSR_data_cleaned.csv      ← Dataset original (~188k filas)
  msr_pipeline.csv          ← Dataset procesado (~377k filas)
  validation_report.txt     ← Reporte de validación
  predictions.csv           ← Predicciones batch

models/
  security_classifier_msr.joblib  ← Modelo entrenado
  training_report.txt             ← Métricas de entrenamiento
  plots/
    confusion_matrix.png          ← Matriz de confusión
    feature_importance.png        ← Features importantes

scripts/
  run_pipeline_completo.py   ← 🌟 Pipeline automatizado
  proceso_msr.py             ← Procesar MSR
  validar_datos.py           ← Validar dataset
  entrenar_modelo.py         ← Entrenar modelo
  inferencia_pruebas.py      ← Inferencia y pruebas
```

---

## 🔧 Instalación

```powershell
# 1. Entorno virtual (opcional pero recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Dependencias
pip install -r requirements.txt

# 3. Instalar paquete
pip install -e .
```

---

## 📈 Métricas Esperadas

```
Accuracy:   0.85-0.90
Precision:  0.75-0.85
Recall:     0.70-0.80
F1-Score:   0.70-0.80
```

---

## 💡 Opciones Comunes

### Procesar con rutas personalizadas
```powershell
python scripts\proceso_msr.py --input data\custom.csv --output data\output.csv
```

### Entrenar con configuración personalizada
```powershell
python scripts\entrenar_modelo.py --dataset data\msr_pipeline.csv --model models\custom.joblib --test-size 0.3
```

### Entrenar sin gráficos (más rápido)
```powershell
python scripts\entrenar_modelo.py --no-plots
```

### Analizar con lenguaje específico
```powershell
python scripts\inferencia_pruebas.py --file code.cpp --language C++
```

---

## 🆘 Problemas Comunes

### Error: archivo no encontrado
```powershell
# Verificar ubicación
dir data\MSR_data_cleaned.csv
```

### Error: módulo no encontrado
```powershell
# Reinstalar dependencias
pip install -r requirements.txt
pip install -e .
```

### Proceso muy lento
```powershell
# Usar sin gráficos
python scripts\entrenar_modelo.py --no-plots
```

---

## 📚 Documentación Completa

Ver [GUIA_MSR_DATA.md](GUIA_MSR_DATA.md) para:
- Explicación detallada de cada paso
- Solución de problemas
- Configuración avanzada
- Integración CI/CD
- Ejemplos completos

---

## ✅ Flujo de Trabajo Típico

```powershell
# Primer uso
python scripts\run_pipeline_completo.py

# Análisis posterior de archivos
python scripts\inferencia_pruebas.py --file mi_codigo.c

# Re-entrenar con ajustes
python scripts\entrenar_modelo.py --test-size 0.25
```

---

**Última actualización:** Diciembre 2025
