from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import requests
import os

# Inicializar la app
app = FastAPI()

# Cargar modelo y datos locales
modelo = joblib.load("modelo.pkl")
vectorizador = joblib.load("vectorizador.pkl")
df_libros = pd.read_csv("libros.csv")

# API Key de Google Books (pon tu key aquí o usa una variable de entorno)
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "AQUI_TU_API_KEY")


class LibroFavorito(BaseModel):
    titulo: str
    descripcion: str


class SolicitudRecomendacion(BaseModel):
    favoritos: List[LibroFavorito]


def buscar_google_books(query: str, max_results: int = 5):
    """Busca libros en Google Books sin necesidad de API Key."""
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    libros_google = []
    if "items" not in data:
        return []

    for item in data["items"]:
        volumen = item.get("volumeInfo", {})
        libros_google.append({
            "id": item.get("id", ""),
            "titulo": volumen.get("title", "Título desconocido"),
            "descripcion": volumen.get("description", "Sin descripción disponible"),
            "texto": (volumen.get("title", "") + " " + volumen.get("description", ""))
        })
    return libros_google

@app.post("/recomendar")
def recomendar_libros(solicitud: SolicitudRecomendacion):
    if len(solicitud.favoritos) == 0:
        return {"error": "No se enviaron libros favoritos"}

    # Texto combinado de favoritos
    textos = [f"{libro.titulo} {libro.descripcion}" for libro in solicitud.favoritos]
    vectores = vectorizador.transform(textos)
    vector_promedio = np.mean(vectores.toarray(), axis=0).reshape(1, -1)

    # Recomendaciones locales
    k = min(5, len(df_libros))
    distancias, indices = modelo.kneighbors(vector_promedio, n_neighbors=k)
    recomendaciones_locales = df_libros.iloc[indices[0]].to_dict(orient="records")

    # Recomendaciones Google Books
    query_google = solicitud.favoritos[0].titulo
    recomendaciones_google = buscar_google_books(query_google, max_results=5)
    
    ids_Locales = [item["google_id"] for item in recomendaciones_locales]
    ids_google = [item["id"] for item in recomendaciones_google]


    return {
        "recomendaciones_locales": ids_Locales,
        "recomendaciones_google": ids_google
    }
