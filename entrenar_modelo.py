#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# ---------- NLTK stopwords (es) ----------
import nltk 
from nltk.corpus import stopwords

# Descarga el corpus de stopwords si no está disponible
try:
    _ = stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

spanish_stopwords = stopwords.words('spanish')
# -----------------------------------------

# 1) Cargar dataset
#    Asegúrate de tener el CSV en el mismo directorio:
#    libros_cundinamarca.csv con las columnas: id, google_id, titulo, descripcion, texto
df = pd.read_csv("libros_cundinamarca.csv")

# 2) Construir el texto a vectorizar (une todo lo útil)
df['texto_full'] = (
    df.get('titulo', '').fillna('') + " " +
    df.get('descripcion', '').fillna('') + " " +
    df.get('texto', '').fillna('')
)

# 3) Vectorizador TF-IDF con stopwords en español
vectorizador = TfidfVectorizer(stop_words=spanish_stopwords)
X = vectorizador.fit_transform(df['texto_full'])

# 4) Modelo KNN (coseno). Evita pedir más vecinos que registros
k = min(10, len(df))
modelo = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
modelo.fit(X)

# 5) Guardar artefactos para FastAPI
joblib.dump(modelo, "modelo.pkl")
joblib.dump(vectorizador, "vectorizador.pkl")
df.to_csv("libros.csv", index=False)

print(f"✅ Modelo entrenado con {len(df)} libros y k={k}. Archivos guardados: modelo.pkl, vectorizador.pkl, libros.csv")
