# Predicción de Popularidad de Canciones en Spotify

**Evaluación Parcial N°1 — MLY1101 Machine Learning**
**Caso C: Inteligencia musical y predicción de popularidad de canciones**

## 1. Descripción del problema de negocio

Una distribuidora/plataforma musical quiere anticipar qué tan popular será una canción a partir de sus características de audio y metadatos, para priorizar qué temas promocionar en playlists y campañas, optimizando el uso de su presupuesto de marketing.

## 2. Objetivos del proyecto

Desarrollar un modelo de Machine Learning que prediga el score de popularidad (`popularity`, escala 0-100) de una canción a partir de sus atributos de audio y de género, para apoyar decisiones de promoción musical.

## 3. Definición de KPIs

- **Error de predicción del modelo:** RMSE / MAE sobre `popularity`, para medir qué tan preciso es el modelo.
- **% de canciones "top" correctamente identificadas:** proporción de canciones con `popularity > 70` que el modelo logra detectar, como medida de utilidad práctica para priorizar promoción.

## 4. Descripción de las fuentes de datos

- **Fuente:** [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), publicado por MaharshiPandya.
- **Volumen original:** 114.000 canciones, 21 columnas, 125 géneros musicales.
- **Variables principales:** metadatos (artista, álbum, nombre, género) y audio features calculadas por Spotify (danceability, energy, loudness, valence, tempo, acousticness, instrumentalness, speechiness, liveness, key, mode, time_signature, duration_ms, explicit).
- **Variable objetivo:** `popularity` (0-100).
- **Herramienta colaborativa:** repositorio en GitHub para versionar el proyecto.

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Calidad de datos y limpieza

- Nulos mínimos: solo 1 fila sin `artists`, `album_name` y `track_name` (eliminada).
- Se eliminó la columna índice sobrante (`Unnamed: 0`), lo que reveló **450 filas 100% duplicadas** (eliminadas).
- Se detectaron **40.900 filas con `track_id` repetido**, de las cuales la gran mayoría (16.299 grupos) corresponden a la **misma canción catalogada bajo distintos géneros** — se conservan por ser observaciones canción-género válidas, no errores. Solo 342 grupos eran duplicados reales con el mismo género (absorbidos en la limpieza de duplicados exactos).
- Se identificó 1 canción con `duration_ms = 0` (valor inválido), eliminada.
- **Dataset final:** 113.549 filas x 20 columnas (`data/processed/spotify_tracks_clean.csv`).

### 5.2 Hallazgos del EDA

- La distribución de `popularity` está sesgada hacia valores bajos-medios, con ~14% de canciones en `popularity = 0`.
- Las audio features individuales (danceability, energy, loudness, valence, tempo, etc.) tienen **correlación muy baja** con `popularity` (todas por debajo de 0.1 en valor absoluto).
- El **género musical (`track_genre`) sí influye fuertemente**: géneros como `pop-film` y `k-pop` promedian ~55-60 de popularidad, mientras que géneros como `iranian` o `romance` promedian menos de 10.
- Las canciones explícitas muestran una popularidad promedio levemente superior a las no explícitas.

Ver notebooks `01_preparacion_datos.ipynb` y `02_eda.ipynb` para el detalle completo, y `images/` para los gráficos generados.

### 5.3 Evaluación de sesgos, ética y privacidad

- **Privacidad:** el dataset no contiene datos personales de usuarios, solo metadatos públicos de canciones y métricas agregadas. Riesgo de privacidad bajo.
- **Sesgo de género musical:** un modelo entrenado con estos datos podría favorecer patrones de géneros mainstream y subestimar canciones de géneros de nicho.
- **Sesgo "long tail":** la concentración de popularidad en pocas canciones puede sesgar el modelo hacia predecir valores bajos por defecto.
- **Limitación:** no existen variables temporales ni de contexto (fecha de lanzamiento, marketing, ubicación), por lo que el modelo solo puede capturar la relación entre audio features/género y popularidad, no factores externos.
- **Mitigación:** el modelo debe usarse como apoyo a la decisión de promoción musical, no como reemplazo del criterio experto (A&R).

## 6. Metodología utilizada (CRISP-DM)

Este proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

1. **Comprensión del negocio:** definición del problema, objetivos y KPIs (secciones 1-3).
2. **Comprensión de los datos:** exploración inicial del dataset, tipos de datos, nulos y duplicados (`01_preparacion_datos.ipynb`).
3. **Preparación de los datos:** limpieza y tratamiento de duplicados/valores inválidos (`01_preparacion_datos.ipynb`).
4. **Análisis exploratorio (EDA):** identificación de patrones y relación entre variables y `popularity` (`02_eda.ipynb`).
5. **Evaluación de sesgos y ética:** análisis de limitaciones y riesgos del dataset (`02_eda.ipynb`, sección 4).

## Estructura del proyecto

```
ev1_machine_learning/
├── data/
│   ├── raw/                  # Dataset original (dataset.csv)
│   └── processed/            # Dataset limpio (spotify_tracks_clean.csv)
├── notebooks/
│   ├── 01_preparacion_datos.ipynb
│   └── 02_eda.ipynb
├── images/                   # Gráficos generados en el EDA
├── models/                   # (siguiente etapa del proyecto)
└── README.md
```
