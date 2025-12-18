# 📁 ÍNDICE MAESTRO DEL PROYECTO

## 🎯 Objetivo del Proyecto

Sistema completo de detección de vulnerabilidades en código fuente usando Machine Learning (RandomForest) con dataset MSR de ~188k vulnerabilidades reales en C/C++.

---

## 📚 Documentación

### 🌟 Para Empezar

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** ← ⭐ **EMPIEZA AQUÍ**
   - Comandos esenciales
   - Referencia rápida
   - Solución de problemas comunes

2. **[GUIA_MSR_DATA.md](GUIA_MSR_DATA.md)** ← 📘 Guía completa
   - Documentación detallada
   - Explicación paso a paso
   - Configuración avanzada
   - Integración CI/CD

3. **[README.md](README.md)** ← Información general
   - Descripción del proyecto
   - Opciones de uso
   - Arquitectura del pipeline

---

## 🛠️ Scripts Principales

### Pipeline Automatizado (Recomendado)
```powershell
python scripts\run_pipeline_completo.py
```

### Scripts Individuales

| Script | Propósito | Comando |
|--------|-----------|---------|
| **proceso_msr.py** | Procesar MSR_data_cleaned.csv | `python scripts\proceso_msr.py` |
| **validar_datos.py** | Validar dataset | `python scripts\validar_datos.py data\msr_pipeline.csv` |
| **entrenar_modelo.py** | Entrenar modelo | `python scripts\entrenar_modelo.py` |
| **inferencia_pruebas.py** | Probar modelo | `python scripts\inferencia_pruebas.py --test-examples` |

### Scripts Auxiliares

| Script | Propósito |
|--------|-----------|
| **analyze_msr_csv.py** | Analizar estructura del CSV original |
| **auto_run.py** | Automatización personalizada |
| **optimize.py** | Optimización de hiperparámetros |

---

## 📂 Estructura del Proyecto

```
Proyecto_UII_DSS/
│
├── 📄 QUICK_REFERENCE.md          ← ⭐ EMPIEZA AQUÍ
├── 📄 GUIA_MSR_DATA.md            ← Documentación completa
├── 📄 README.md                   ← Información general
├── 📄 INDEX.md                    ← Este archivo
│
├── 📁 data/                       ← Datasets
│   ├── MSR_data_cleaned.csv       ← Dataset original (~188k filas)
│   ├── msr_pipeline.csv           ← Dataset procesado (~377k filas)
│   ├── validation_report.txt      ← Reporte de validación
│   └── predictions.csv            ← Predicciones (generado)
│
├── 📁 models/                     ← Modelos entrenados
│   ├── security_classifier_msr.joblib  ← Modelo principal
│   ├── training_report.txt             ← Métricas de entrenamiento
│   └── plots/                          ← Visualizaciones
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── 📁 scripts/                    ← Scripts principales
│   ├── run_pipeline_completo.py   ← 🌟 Pipeline automatizado
│   ├── proceso_msr.py             ← Procesar MSR
│   ├── validar_datos.py           ← Validar dataset
│   ├── entrenar_modelo.py         ← Entrenar modelo
│   ├── inferencia_pruebas.py      ← Inferencia
│   └── analyze_msr_csv.py         ← Análisis CSV
│
├── 📁 src/secure_pipeline/        ← Módulos del paquete
│   ├── __init__.py
│   ├── data.py                    ← Carga de datos
│   ├── features.py                ← Extracción de features
│   ├── train.py                   ← Entrenamiento
│   ├── infer.py                   ← Inferencia
│   └── convert_bigvul.py          ← Conversión BigVul
│
├── 📁 logs/                       ← Logs del sistema
├── 📄 requirements.txt            ← Dependencias Python
├── 📄 pyproject.toml              ← Configuración del paquete
├── 🐍 demo_safe.c                 ← Ejemplo código seguro
└── 🐍 demo_unsafe.c               ← Ejemplo código vulnerable
```

---

## 🚀 Inicio Rápido (3 Pasos)

### 1️⃣ Instalar
```powershell
pip install -r requirements.txt
pip install -e .
```

### 2️⃣ Ejecutar
```powershell
python scripts\run_pipeline_completo.py
```

### 3️⃣ Probar
```powershell
python scripts\inferencia_pruebas.py --file demo_unsafe.c
```

---

## 📊 Flujo de Datos

```
MSR_data_cleaned.csv (188k filas)
         ↓
   [proceso_msr.py]
         ↓
msr_pipeline.csv (377k filas)
         ↓
   [validar_datos.py]
         ↓
   [entrenar_modelo.py]
         ↓
security_classifier_msr.joblib
         ↓
   [inferencia_pruebas.py]
         ↓
    Predicciones
```

---

## 🎓 Casos de Uso

### 1. Análisis de un archivo
```powershell
python scripts\inferencia_pruebas.py --file mi_codigo.c
```

### 2. Modo interactivo
```powershell
python scripts\inferencia_pruebas.py --interactive
```

### 3. Análisis batch
```powershell
python scripts\inferencia_pruebas.py --dataset data\msr_pipeline.csv --sample 1000
```

### 4. Re-entrenar modelo
```powershell
python scripts\entrenar_modelo.py --test-size 0.25
```

### 5. Validar antes de usar
```powershell
python scripts\validar_datos.py data\msr_pipeline.csv --report
```

---

## 🔑 Características Clave

✅ **Pipeline Automatizado** - Un comando ejecuta todo  
✅ **Validación Completa** - Verifica integridad de datos  
✅ **Métricas Detalladas** - Accuracy, Precision, Recall, F1, AUC-ROC  
✅ **Visualizaciones** - Matrices de confusión, feature importance  
✅ **Reportes Automáticos** - TXT con todos los resultados  
✅ **Modo Interactivo** - Prueba código en tiempo real  
✅ **Análisis Batch** - Procesa múltiples archivos  
✅ **Desbalance Manejado** - class_weight='balanced_subsample'  
✅ **Cross-Validation** - 5-fold para validación robusta  
✅ **Múltiples Lenguajes** - C, C++, Python, Java

---

## 📈 Métricas del Modelo

Con ~377k muestras de entrenamiento:

```
Accuracy:        0.85-0.90
Precision:       0.75-0.85
Recall:          0.70-0.80
F1-Score:        0.70-0.80
Tiempo entrenamiento: 30-60 min (depende del hardware)
```

---

## 🆘 Ayuda Rápida

| Problema | Solución |
|----------|----------|
| Archivo no encontrado | Verifica `data\MSR_data_cleaned.csv` |
| Módulo no encontrado | `pip install -r requirements.txt` |
| Proceso lento | Usa `--no-plots` al entrenar |
| Error de memoria | Cierra otras aplicaciones |
| Advertencia DtypeWarning | Es normal, se maneja automáticamente |

---

## 📞 Soporte

1. **Referencia Rápida:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Guía Completa:** [GUIA_MSR_DATA.md](GUIA_MSR_DATA.md)
3. **Logs:** Revisar carpeta `logs/`
4. **Reportes:** Revisar `models/training_report.txt` y `data/validation_report.txt`

---

## ✅ Checklist de Configuración

- [ ] Python 3.8+ instalado
- [ ] `pip install -r requirements.txt` ejecutado
- [ ] `pip install -e .` ejecutado
- [ ] `MSR_data_cleaned.csv` en `data/`
- [ ] Pipeline ejecutado: `python scripts\run_pipeline_completo.py`
- [ ] Modelo generado en `models/security_classifier_msr.joblib`
- [ ] Pruebas funcionando: `python scripts\inferencia_pruebas.py --test-examples`

---

## 🎯 Próximos Pasos

Después de completar el setup:

1. ✅ **Revisar métricas** en `models/training_report.txt`
2. ✅ **Ver gráficos** en `models/plots/`
3. ✅ **Probar con ejemplos**: `demo_safe.c` y `demo_unsafe.c`
4. ✅ **Modo interactivo** para pruebas rápidas
5. ✅ **Integrar en CI/CD** (ver guía completa)

---

**Proyecto:** Pipeline CI/CD con Clasificador de Vulnerabilidades  
**Dataset:** MSR_data_cleaned.csv (~188k vulnerabilidades C/C++)  
**Modelo:** RandomForest optimizado para desbalance  
**Última actualización:** Diciembre 2025

---

## 🌟 Archivo Recomendado para Empezar

### 👉 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ← COMIENZA AQUÍ
