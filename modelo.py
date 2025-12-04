import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# 1. CARGAR DATASET
data = pd.read_csv("CVEfixes.csv")

# 2. LIMPIEZA BÁSICA
data = data.dropna()

# 3. CODIFICAR LA COLUMNA "language"
encoder = LabelEncoder()
data["language"] = encoder.fit_transform(data["language"])

# 4. VARIABLES DE ENTRADA Y SALIDA
X_text = data["code"]          # Código fuente (texto)
X_lang = data["language"]     # Lenguaje (numérico)
y = data["safety"]             # Etiqueta: vulnerable / safe

# 5. CONVERTIR TEXTO A TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X_code = tfidf.fit_transform(X_text)

# 6. COMBINAR TF-IDF + LENGUAJE
X_final = np.hstack((X_code.toarray(), X_lang.values.reshape(-1, 1)))

# 7. DIVISIÓN DE DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=42, stratify=y
)

# 8. ENTRENAMIENTO DEL MODELO
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 9. PREDICCIÓN
y_pred = model.predict(X_test)

# 10. EVALUACIÓN
print("\n================ RESULTADOS DEL MODELO ================\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))
