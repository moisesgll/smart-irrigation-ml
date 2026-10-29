"""
Agricultura Inteligente con Machine Learning 

Proyecto EDUCATIVO. Un árbol de decisión predice qué zonas de un terreno de
8 zonas (4 maíz + 4 arroz) necesitan riego, y una simulación de 7 días
mide el consumo de agua y la salud de las zonas con un máximo de 4 válvulas
activas por día.

USO
    python agricultura_inteligente.py                    # --datos auto
    python agricultura_inteligente.py --datos sintetico  # datos sintéticos
    python agricultura_inteligente.py --datos kaggle     # exige el CSV
    python agricultura_inteligente.py --salida graficos  # guarda los PNG

PROCEDENCIA Y SIGNIFICADO DE LOS DATOS 
  * SINTÉTICO: las etiquetas 'regar' salen de una regla heurística explícita
    con ruido (ver _etiquetar_sintetico). Representan "necesidad de riego
    según esa regla", NO una medición agronómica. Es el modo recomendado
    para esta demostración educativa.
  * KAGGLE (experimental / indirecto): usa 'Smart_Farming_Crop_Yield_2024.csv'
    si está en la carpeta. Incluye los cultivos Maize (maíz) y Rice (arroz),
    pero su columna 'irrigation_type' describe el tipo de riego usado, no si
    el suelo necesitaba agua. Aquí 'regar' = "hubo algún tipo de riego"
    (historial). Las métricas de ese modo miden cuánto se parece el árbol a
    ese historial de riego, no la necesidad real de agua del maíz o el
    arroz. El texto "None" de
    'irrigation_type' es una categoría válida (sin riego explícito → regar=0);
    una celda vacía o ausente es una etiqueta DESCONOCIDA y esa fila se excluye
    del entrenamiento (nunca se etiqueta como regar=0).
  * La caché 'riego_dataset.csv' va acompañada de 'riego_dataset.meta.json'
    con su origen. Una caché sin metadatos se ignora (origen desconocido).

SIMPLIFICACIONES DE LA SIMULACIÓN (intencionales)
  Clima aleatorio (temperatura y humedad del aire), lluvia probabilística
  (PROB_LLUVIA), incremento fijo de humedad por riego, decaimiento fijo por
  día sin riego (ambos distintos según cultivo) y sin modelo de
  evapotranspiración, tipo de suelo, drenaje ni caudal real. Los parámetros
  de humedad y los umbrales son valores del proyecto, no valores agronómicos
  validados.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

# ─── MODELO (Árbol de Decisión) ────────────────────────────────────────────
DECISION_TREE_MAX_DEPTH = 6
DECISION_TREE_MIN_LEAF  = 5
RANDOM_STATE            = 42
TEST_SIZE               = 0.2

# ─── DATASET ────────────────────────────────────────────────────────────────
# Nombre del CSV de Kaggle tal cual se descarga (no renombrar).
KAGGLE_CSV_PATH = "Smart_Farming_Crop_Yield_2024.csv"
DATASET_PATH    = "riego_dataset.csv"
N_SAMPLES_SYNTH = 1200   # tamaño del dataset sintético de RESPALDO únicamente

MODO_DATOS      = "auto"
MODOS_DATOS     = ("auto", "sintetico", "kaggle")

# Cadenas que se leen como valor AUSENTE en el CSV de Kaggle. Son las que pandas
# reconoce por defecto SALVO "None": en 'irrigation_type', "None" es una
# categoría con significado propio (no se aplicó riego), no un dato faltante.
NA_VALUES_KAGGLE = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN",
                    "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN",
                    "n/a", "nan", "null"]

# Esquema que debe cumplir cualquier dataset antes de entrenar.
COLUMNAS_DATASET      = ["humedad_suelo", "temperatura", "humedad_aire", "cultivo", "regar"]
MIN_REGISTROS_DATASET = 50   # mínimo razonable (experimental) para entrenar/evaluar
CACHE_VERSION         = 3    # sube este número si cambia el formato de la caché

# Cultivos del dataset de Kaggle (Wheat, Rice, Maize, Cotton, Soybean) que usa
# el proyecto, con el nombre y el código 0/1 que emplea el modelo:
# Maize -> "maiz" (código 0) y Rice -> "arroz" (código 1).
# Puedes cambiar el par si prefieres otros cultivos del dataset.
MAPEO_CULTIVOS_KAGGLE = {
    "Maize": "maiz",
    "Rice":  "arroz",
}

# Codificación numérica del cultivo para el modelo (0/1)
CODIGO_CULTIVO = {"maiz": 0, "arroz": 1}

# ─── TERRENO (8 zonas) ──────────────────────────────────────────────────────
# Layout:
#   M1 M2 M3 M4
#   A1 A2 A3 A4
# M -> maíz, A -> arroz
ZONAS = [
    {"id": "M1", "cultivo": "maiz"},
    {"id": "M2", "cultivo": "maiz"},
    {"id": "M3", "cultivo": "maiz"},
    {"id": "M4", "cultivo": "maiz"},
    {"id": "A1", "cultivo": "arroz"},
    {"id": "A2", "cultivo": "arroz"},
    {"id": "A3", "cultivo": "arroz"},
    {"id": "A4", "cultivo": "arroz"},
]

# Humedad inicial por zona (%) — punto de partida de la simulación.
HUMEDAD_INICIAL = {
    "M1": 55, "M2": 48, "M3": 60, "M4": 35,
    "A1": 42, "A2": 50, "A3": 30, "A4": 58,
}

# ─── REGLAS DE SIMULACIÓN ───────────────────────────────────────────────────
DIAS_SIMULACION = 7

# Decaimiento/incremento DIFERENCIADO por cultivo (Problema 3).
# El arroz (código 1) pierde y gana humedad más rápido que el maíz (código 0).
# Son valores del proyecto , no valores agronómicos validados.
REGLAS_POR_CULTIVO = {
    "maiz": {
        "decaimiento_sin_riego": 4,
        "incremento_con_riego":  14,
    },
    "arroz": {
        "decaimiento_sin_riego": 6,
        "incremento_con_riego":  16,
    },
}

LITROS_POR_VALVULA    = 10    # L que entrega cada válvula activada (fijo, regla del proyecto)

PROB_LLUVIA           = 0.05  # 5%
INCREMENTO_LLUVIA     = 10    # Ht+1 = Ht + 10 si llueve
UMBRAL_SATURACION_LLUVIA = 85 

MAX_ZONAS_POR_DIA     = 4     # restricción de válvulas simultáneas/día

HUMEDAD_MIN           = 0
HUMEDAD_MAX           = 100

# Ponderación para la prioridad de riego:
#   score = PESO_PROBABILIDAD * probabilidad_del_modelo
#         + (1 - PESO_PROBABILIDAD) * criticidad_por_humedad
# criticidad_por_humedad = (100 - humedad) / 100  (más seca = más crítica)
PESO_PROBABILIDAD     = 0.7

# ─── EVALUACIÓN / MÉTRICAS DE ÉXITO ────────────────────────────────────────
CONSUMO_BASE_L        = 560   # litros consumidos sin optimización (referencia)
CONSUMO_OBJETIVO_L    = 420   # meta: <= 420 L en la semana
AHORRO_MINIMO         = 0.25  # Objetivo >= 25% de ahorro respecto al base
UMBRAL_HUMEDAD_SANA   = 40    # humedad saludable mínima (%)
PORC_ZONAS_SANAS_MIN  = 0.80  # >= 80% de las zonas deben mantenerse sanas

# Regla de seguridad (independiente del modelo): una zona cuya humedad al
# inicio del día está POR DEBAJO de este valor es "prioritaria" y se riega
# aunque el árbol no lo pida (respetando MAX_ZONAS_POR_DIA). Por defecto
# reutiliza UMBRAL_HUMEDAD_SANA. Es un parámetro EXPERIMENTAL del proyecto,
# no un umbral agronómico validado.
UMBRAL_HUMEDAD_SEGURIDAD = UMBRAL_HUMEDAD_SANA

# Tamaño mínimo de la clase minoritaria a partir del cual las métricas se
# consideran razonablemente fiables (experimental). Por debajo se advierte.
MIN_MUESTRAS_CLASE_FIABLE = 10


# ════════════════════════════════════════════════════════════════════════════
# 2. DATASET 
# ════════════════════════════════════════════════════════════════════════════

def _normalizar_columnas_kaggle(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el CSV crudo de Kaggle (Smart_Farming_Crop_Yield_2024.csv) y lo
    transforma al esquema que usa nuestro modelo:

        humedad_suelo, temperatura, humedad_aire, cultivo, regar

    Pasos:
      1. Detecta las columnas relevantes por nombre (tolerante a mayúsculas
         y variaciones de '%').
      2. Filtra solo las filas cuyo crop_type esté en MAPEO_CULTIVOS_KAGGLE.
      3. Les asigna el nombre y el código 0/1 de 'maiz' / 'arroz'.
      4. Convierte 'irrigation_type' a la etiqueta binaria 'regar':
             'None'          -> 0 (el texto "None": no se aplicó riego)
             otro texto      -> 1 (se aplicó algún tipo de riego)
             vacío / ausente -> etiqueta DESCONOCIDA: la fila se EXCLUYE
                                (no se etiqueta como 0 ni como 1)
         OJO: esto es HISTORIAL DE RIEGO OBSERVADO, no necesidad real de agua.
         Un registro con riego no implica que el suelo lo necesitara, ni uno
         sin riego que no lo necesitara.
    """
    cols = {c.lower().strip(): c for c in df_raw.columns}

    def buscar(*candidatos):
        for cand in candidatos:
            if cand in cols:
                return cols[cand]
        for key, original in cols.items():
            if any(cand.replace("_", "") in key.replace("_", "") for cand in candidatos):
                return original
        raise ValueError(
            f"El CSV de Kaggle no tiene ninguna columna compatible con {candidatos}. "
            f"Columnas encontradas: {list(df_raw.columns)}")

    col_moisture = buscar("soil_moisture_%", "soil_moisture", "soilmoisture")
    col_temp     = buscar("temperature_c", "temperature", "temp_c")
    col_humidity = buscar("humidity_%", "humidity", "air_humidity")
    col_crop     = buscar("crop_type", "crop")
    col_irrig    = buscar("irrigation_type", "irrigation")

    df = pd.DataFrame({
        "humedad_suelo": pd.to_numeric(df_raw[col_moisture], errors="coerce"),
        "temperatura":   pd.to_numeric(df_raw[col_temp], errors="coerce"),
        "humedad_aire":  pd.to_numeric(df_raw[col_humidity], errors="coerce"),
        "crop_type_raw": df_raw[col_crop].astype(str).str.strip(),
        "irrigation_raw": df_raw[col_irrig].astype("string").str.strip().fillna(""),
    })

    df = df[df["crop_type_raw"].isin(MAPEO_CULTIVOS_KAGGLE.keys())].copy()
    df["cultivo_nombre"] = df["crop_type_raw"].map(MAPEO_CULTIVOS_KAGGLE)
    df["cultivo"] = df["cultivo_nombre"].map(CODIGO_CULTIVO)

    # Etiqueta desconocida (celda vacía/ausente): se excluye del entrenamiento
    sin_etiqueta = df["irrigation_raw"] == ""
    n_sin_etiqueta = int(sin_etiqueta.sum())
    if n_sin_etiqueta and n_sin_etiqueta == len(df):
        raise ValueError(
            f"Las {n_sin_etiqueta} filas de los cultivos {list(MAPEO_CULTIVOS_KAGGLE)} no "
            f"tienen etiqueta de riego válida ('{col_irrig}' vacío o ausente).")
    df = df[~sin_etiqueta].copy()

    # Etiqueta binaria: historial de riego observado (NO necesidad real de agua)
    df["regar"] = (df["irrigation_raw"].str.lower() != "none").astype(int)

    df = df.dropna(subset=["humedad_suelo", "temperatura", "humedad_aire", "cultivo"])
    df = df[["humedad_suelo", "temperatura", "humedad_aire", "cultivo", "regar"]]
    df = df.reset_index(drop=True)
    df.attrs["n_sin_etiqueta"] = n_sin_etiqueta

    return df


def _leer_csv_kaggle(ruta) -> pd.DataFrame:
    """
    Lee el CSV de Kaggle conservando el texto "None" (categoría válida de
    'irrigation_type') y reconociendo como AUSENTES las celdas vacías y los
    marcadores habituales (NA, NaN, null...; ver NA_VALUES_KAGGLE). Las columnas
    numéricas se siguen convirtiendo con pd.to_numeric en la normalización.
    """
    return pd.read_csv(ruta, keep_default_na=False, na_values=NA_VALUES_KAGGLE)


def _etiquetar_sintetico(humedad_suelo, temperatura, humedad_aire, cultivo):
    """
    ETIQUETAS SINTÉTICAS (modo "sintetico" o respaldo si no hay CSV de Kaggle).
    Regla heurística explícita con ruido:

        score = (40 - humedad_suelo) * sensibilidad     # 1.15 arroz, 1.0 maíz
              + 0.8 * (temperatura - 22)
              + 0.3 * (50 - humedad_aire)
        regar = 1 si score + N(0, 6) > 0, si no 0

    Es REPRODUCIBLE (usa np.random.seed(seed) en generar_dataset_sintetico)
    y NO es una medición agronómica: el árbol solo aprende esta regla.
    """
    sensibilidad = np.where(cultivo == 1, 1.15, 1.0)  # arroz = 1, maíz = 0
    score = (
        (40 - humedad_suelo) * 1.0 * sensibilidad
        + (temperatura - 22) * 0.8
        + (50 - humedad_aire) * 0.3
    )
    ruido = np.random.normal(0, 6, size=len(score))
    return ((score + ruido) > 0).astype(int)


def generar_dataset_sintetico(n_samples: int = N_SAMPLES_SYNTH,
                               seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Dataset SINTÉTICO (etiquetas por regla explícita; ver _etiquetar_sintetico)."""
    rng = np.random.default_rng(seed)

    humedad_suelo = rng.uniform(10, 90, n_samples)
    temperatura   = rng.normal(22, 6, n_samples).clip(5, 40)
    humedad_aire  = rng.uniform(20, 95, n_samples)
    cultivo       = rng.integers(0, 2, n_samples)  # 0=maiz, 1=arroz

    np.random.seed(seed)
    regar = _etiquetar_sintetico(humedad_suelo, temperatura, humedad_aire, cultivo)

    return pd.DataFrame({
        "humedad_suelo": humedad_suelo.round(1),
        "temperatura":   temperatura.round(1),
        "humedad_aire":  humedad_aire.round(1),
        "cultivo":       cultivo,
        "regar":         regar,
    })


def validar_dataset(df, min_filas: int = 1, nombre: str = "El dataset") -> None:
    """
    Comprueba que el dataset sea utilizable. Lanza ValueError con un mensaje
    claro si está vacío, le faltan columnas, tiene muy pocas filas, valores
    nulos / no numéricos / infinitos, o etiquetas y cultivos fuera de {0, 1}.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"{nombre} está vacío (0 filas).")
    faltan = [c for c in COLUMNAS_DATASET if c not in df.columns]
    if faltan:
        raise ValueError(f"{nombre} es incompatible: faltan las columnas {faltan}. "
                         f"Se esperaban {COLUMNAS_DATASET}.")
    if len(df) < min_filas:
        raise ValueError(f"{nombre} tiene solo {len(df)} filas (mínimo {min_filas}).")
    for col in COLUMNAS_DATASET:
        serie = pd.to_numeric(df[col], errors="coerce")
        n_malos = int(serie.isna().sum())
        if n_malos:
            raise ValueError(f"{nombre}: la columna '{col}' tiene {n_malos} valores "
                             f"nulos o no numéricos.")
        if not np.isfinite(serie).all():
            raise ValueError(f"{nombre}: la columna '{col}' contiene valores infinitos.")
    for col in ("regar", "cultivo"):
        invalidos = set(pd.to_numeric(df[col]).unique()) - {0, 1}
        if invalidos:
            raise ValueError(f"{nombre}: '{col}' solo admite 0/1 y contiene {sorted(invalidos)}.")


_SIGNIFICADO_ETIQUETA = {
    "sintetico": "necesidad de riego según una regla heurística explícita con ruido "
                 "(NO es una medición agronómica)",
    "kaggle":    "historial de riego observado (irrigation_type != 'None'); NO es la "
                 "necesidad real de agua",
}


def _marcar_fuente(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Anota en df.attrs de dónde vienen los datos y qué significa 'regar'."""
    df.attrs["fuente"] = fuente
    df.attrs["significado_regar"] = _SIGNIFICADO_ETIQUETA[
        "kaggle" if fuente.startswith("kaggle") else "sintetico"]
    return df


def _ruta_meta(cache_file: Path) -> Path:
    return cache_file.with_suffix(".meta.json")


def _guardar_cache(df: pd.DataFrame, cache_file: Path, origen: str, **extra) -> None:
    """Guarda el CSV procesado junto con un .meta.json que declara su origen."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    meta = {"version": CACHE_VERSION, "origen": origen, "n_filas": int(len(df)), **extra}
    _ruta_meta(cache_file).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _leer_cache(cache_file: Path):
    """Devuelve (df, meta) si la caché es válida y de origen conocido; si no, None."""
    if not cache_file.exists():
        return None
    meta_file = _ruta_meta(cache_file)
    if not meta_file.exists():
        print(f"  ⚠️  La caché '{cache_file}' no tiene metadatos (origen desconocido): "
              f"se ignora y no se usará como dataset válido.")
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("version") != CACHE_VERSION:
            raise ValueError("versión de caché distinta")
        if meta.get("origen") not in ("sintetico", "kaggle_procesado"):
            raise ValueError(f"origen no reconocido: {meta.get('origen')!r}")
        df = pd.read_csv(cache_file)
        validar_dataset(df, MIN_REGISTROS_DATASET, "La caché")
        if meta.get("n_filas") != len(df):
            raise ValueError("el número de filas no coincide con sus metadatos")
    except (ValueError, OSError) as e:   # incluye JSON inválido y CSV ilegible
        print(f"  ⚠️  Caché '{cache_file}' no utilizable ({e}): se ignora.")
        return None
    return df, meta


def _diagnostico_kaggle(df: pd.DataFrame) -> None:
    """Muestra qué relación hay realmente entre la etiqueta y la humedad del suelo."""
    print(f"     Rango de humedad_suelo en estos datos: "
          f"{df['humedad_suelo'].min():.1f} – {df['humedad_suelo'].max():.1f}")
    if df["regar"].nunique() == 2 and df["humedad_suelo"].nunique() > 1:
        h0 = df.loc[df["regar"] == 0, "humedad_suelo"].mean()
        h1 = df.loc[df["regar"] == 1, "humedad_suelo"].mean()
        r = df["humedad_suelo"].corr(df["regar"])
        print(f"     Humedad media del suelo: sin riego registrado = {h0:.1f} | "
              f"con riego registrado = {h1:.1f} | correlación con 'regar' = {r:+.2f}")
        print(f"     (Si la correlación es débil, 'regar' no refleja cuándo el suelo "
              f"estaba seco.)")


def _aviso_kaggle(df: pd.DataFrame, origen_txt: str, n_sin_etiqueta=None) -> None:
    print(f"  🧪 MODO EXPERIMENTAL — datos INDIRECTOS de Kaggle ({origen_txt})")
    print(f"     • La etiqueta 'regar' es historial de riego observado, NO necesidad real de agua.")
    print(f"     • Cultivos tomados del dataset: "
          f"{dict(MAPEO_CULTIVOS_KAGGLE)} → códigos {CODIGO_CULTIVO}")
    print(f"     • Las métricas miden qué tanto el árbol imita ese historial, no el riego óptimo.")
    print(f"     Filas tras filtrar: {len(df)}")
    if n_sin_etiqueta is not None:
        print(f"     Filas excluidas por etiqueta de riego desconocida (celda vacía/ausente): "
              f"{n_sin_etiqueta} (no se etiquetaron como 'sin riego').")
    _diagnostico_kaggle(df)
    if df["regar"].nunique() < 2:
        print(f"  ⚠️  ADVERTENCIA: todas las filas tienen 'regar' = {df['regar'].iloc[0]}. "
              f"El árbol no podrá aprender ambas clases. "
              f"Considera cambiar MAPEO_CULTIVOS_KAGGLE a otro par de cultivos.")
    else:
        balance = df["regar"].value_counts(normalize=True)
        if balance.min() < 0.1:
            print(f"  ⚠️  Dataset desbalanceado tras filtrar (clase minoritaria "
                  f"{balance.min():.1%}). Las métricas pueden ser poco confiables.")


def _aviso_sintetico(df: pd.DataFrame, origen_txt: str) -> None:
    print(f"  🧪 DATOS SINTÉTICOS ({origen_txt}) — {len(df)} filas")
    print(f"     • 'regar' = {_SIGNIFICADO_ETIQUETA['sintetico']}.")
    print(f"     • No son datos reales de maíz ni de arroz.")


def load_dataset(kaggle_path: str = KAGGLE_CSV_PATH,
                  cache_path: str = DATASET_PATH,
                  force_regenerate: bool = False,
                  modo: str = MODO_DATOS) -> pd.DataFrame:
    """
    Carga el dataset e informa SIEMPRE de su procedencia (df.attrs['fuente']):
    'kaggle_procesado', 'kaggle_cache', 'sintetico' o 'sintetico_cache'.

    Orden de prioridad (modo="auto"):
      1. Si existe el CSV crudo de Kaggle -> lo procesa y lo cachea
         (datos INDIRECTOS: ver docstring del módulo). Si el CSV existe pero
         no es utilizable, lo dice y cae al respaldo sintético.
      2. Si no, pero hay una caché con metadatos válidos -> la usa.
         Una caché sin metadatos, antigua o inválida se ignora.
      3. Si no hay nada utilizable -> genera dataset sintético (y lo avisa).
    modo="sintetico" ignora el CSV de Kaggle; modo="kaggle" no cae al
    sintético (lanza ValueError). force_regenerate ignora la caché.
    Lanza ValueError con un mensaje claro si el resultado no es utilizable.
    """
    if modo not in MODOS_DATOS:
        raise ValueError(f"modo de datos inválido {modo!r}; usa uno de {MODOS_DATOS}.")
    kaggle_file = Path(kaggle_path)
    cache_file  = Path(cache_path)
    usar_kaggle = modo in ("auto", "kaggle")

    # 1. CSV original de Kaggle
    if usar_kaggle and kaggle_file.exists():
        try:
            df = _normalizar_columnas_kaggle(_leer_csv_kaggle(kaggle_file))
            validar_dataset(df, MIN_REGISTROS_DATASET, "El dataset de Kaggle procesado")
        except ValueError as e:   # incluye CSV vacío, ilegible o incompatible
            if modo == "kaggle":
                raise ValueError(f"No se puede usar '{kaggle_file}': {e}") from e
            print(f"  ❌ '{kaggle_file}' existe pero no es utilizable: {e}")
            print(f"     Se usará el respaldo SINTÉTICO (no son datos reales).")
        else:
            n_sin_etiqueta = df.attrs.get("n_sin_etiqueta")
            _guardar_cache(df, cache_file, "kaggle_procesado", n_sin_etiqueta=n_sin_etiqueta)
            _marcar_fuente(df, "kaggle_procesado")
            _aviso_kaggle(df, f"CSV original '{kaggle_file}', procesado", n_sin_etiqueta)
            return df
    elif modo == "sintetico" and kaggle_file.exists():
        print(f"  ℹ️  Modo 'sintetico': se ignora '{kaggle_file}'.")

    # 2. Caché con origen conocido
    if not force_regenerate:
        cache = _leer_cache(cache_file)
        if cache is not None:
            df, meta = cache
            origen = meta["origen"]
            if origen == "kaggle_procesado" and usar_kaggle and not kaggle_file.exists():
                _marcar_fuente(df, "kaggle_cache")
                _aviso_kaggle(df, f"caché procesada '{cache_file}'", meta.get("n_sin_etiqueta"))
                return df
            if (origen == "sintetico" and modo in ("auto", "sintetico")
                    and meta.get("n_muestras") == N_SAMPLES_SYNTH
                    and meta.get("semilla") == RANDOM_STATE):
                _marcar_fuente(df, "sintetico_cache")
                _aviso_sintetico(df, f"caché '{cache_file}'")
                return df
            print(f"  ℹ️  La caché '{cache_file}' (origen: {origen}) no corresponde al "
                  f"modo/parámetros actuales: se ignora.")

    # 3. Dataset sintético
    if modo == "kaggle":
        raise ValueError(f"Modo 'kaggle': no se encontró '{kaggle_file}' ni una caché "
                         f"procesada de Kaggle válida. Descarga el CSV o usa --datos sintetico.")
    if modo == "auto":
        print(f"  ⚠️  No se usarán datos de Kaggle ('{kaggle_file}' ausente o inutilizable). "
              f"Generando dataset SINTÉTICO.")
        print(f"     Para el modo experimental con Kaggle, descarga "
              f"'Smart_Farming_Crop_Yield_2024.csv' y ponlo en esta carpeta.")
    df = generar_dataset_sintetico()
    validar_dataset(df, MIN_REGISTROS_DATASET, "El dataset sintético")
    _guardar_cache(df, cache_file, "sintetico", n_muestras=N_SAMPLES_SYNTH, semilla=RANDOM_STATE)
    _marcar_fuente(df, "sintetico")
    _aviso_sintetico(df, "generado ahora")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. MODELO — Árbol de Decisión (IrrigationModel)
# ════════════════════════════════════════════════════════════════════════════
#
#     Entradas (X):  [humedad_suelo, temperatura, humedad_aire, cultivo]
#     Salida   (y):  0 = no regar | 1 = activar válvula

class IrrigationModel:
    """Árbol de decisión para predicción de riego por zona."""

    FEATURES = ['humedad_suelo', 'temperatura', 'humedad_aire', 'cultivo']
    TARGET   = 'regar'

    def __init__(
        self,
        max_depth: int = DECISION_TREE_MAX_DEPTH,
        min_samples_leaf: int = DECISION_TREE_MIN_LEAF,
    ) -> None:
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model            = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion='gini',
            random_state=RANDOM_STATE,
        )
        self.is_trained = False
        self.metrics: Dict = {}

    # ─── ENTRENAMIENTO ────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Entrena el modelo con el dataset de irrigación.

        Retorna un dict con accuracy, CV scores, matriz de confusión,
        reporte de clasificación e importancia de variables.
        """
        validar_dataset(df, nombre="El dataset de entrenamiento")
        X = df[self.FEATURES].apply(pd.to_numeric)
        y = pd.to_numeric(df[self.TARGET]).astype(int)

        conteo     = y.value_counts()
        n_clases   = len(conteo)
        min_clase  = int(conteo.min())
        advertencias: List[str] = []

        X_train = X_test = y_train = y_test = None
        if n_clases < 2:
            advertencias.append(
                f"El dataset solo contiene la clase {int(conteo.index[0])}: el modelo "
                f"predecirá siempre lo mismo y NO se puede evaluar (sin test ni CV).")
        elif min_clase < 2:
            advertencias.append(
                "Una clase tiene una sola muestra: no hay partición estratificada "
                "posible, por lo que NO se evalúa (sin test ni CV).")
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y,
                )
            except ValueError as e:
                advertencias.append(
                    f"No fue posible dividir en entrenamiento/prueba ({e}): NO se evalúa.")
        evaluable = X_test is not None

        if evaluable:
            if min_clase < MIN_MUESTRAS_CLASE_FIABLE:
                advertencias.append(
                    f"La clase minoritaria tiene solo {min_clase} muestras "
                    f"(< {MIN_MUESTRAS_CLASE_FIABLE}): las métricas son poco fiables.")
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X, y)
        self.is_trained = True

        accuracy = cm = reporte = y_pred = None
        cv_scores = None
        if evaluable:
            y_pred   = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm       = confusion_matrix(y_test, y_pred, labels=[0, 1])
            reporte  = classification_report(
                y_test, y_pred,
                labels=[0, 1],
                target_names=['No Regar', 'Regar'],
                zero_division=0,
            )

            cv_folds = min(5, min_clase)
            if cv_folds >= 2:
                try:
                    cv_scores = cross_val_score(
                        self.model, X, y, cv=cv_folds, scoring='accuracy'
                    )
                except ValueError as e:
                    advertencias.append(f"No fue posible calcular la validación cruzada ({e}).")

        self.metrics = {
            'accuracy':              accuracy,
            'cv_mean':               cv_scores.mean() if cv_scores is not None else None,
            'cv_std':                cv_scores.std() if cv_scores is not None else None,
            'cv_scores':             cv_scores,
            'confusion_matrix':      cm,
            'classification_report': reporte,
            'feature_importance': (
                dict(zip(self.FEATURES, self.model.feature_importances_))
                if evaluable else None
            ),
            'X_test':  X_test,
            'y_test':  y_test,
            'y_pred':  y_pred,
            'n_train': len(X_train) if evaluable else len(X),
            'n_test':  len(X_test) if evaluable else 0,
            # Metadatos de validez y procedencia
            'evaluacion_valida':   evaluable,
            'cv_valida':           cv_scores is not None,
            'advertencias':        advertencias,
            'distribucion_clases': {int(k): int(v) for k, v in conteo.items()},
            'fuente_datos':        df.attrs.get("fuente", "desconocida"),
            'significado_regar':   df.attrs.get("significado_regar", "no especificado"),
        }

        return self.metrics

    # ─── PREDICCIÓN ───────────────────────────────────────────────────────────

    def predict(self, X) -> np.ndarray:
        """Predice para un DataFrame/array de muestras."""
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a .train() primero.")
        return self.model.predict(X)

    def _row(self, humedad_suelo, temperatura, humedad_aire, cultivo) -> pd.DataFrame:
        return pd.DataFrame(
            [[humedad_suelo, temperatura, humedad_aire, cultivo]],
            columns=self.FEATURES,
        )

    def predict_zone(
        self,
        humedad_suelo: float,
        temperatura:   float,
        humedad_aire:  float,
        cultivo:       int,
    ) -> int:
        """Predice la necesidad de riego para una sola zona (incluye cultivo)."""
        X = self._row(humedad_suelo, temperatura, humedad_aire, cultivo)
        return int(self.predict(X)[0])

    def predict_proba_zone(
        self,
        humedad_suelo: float,
        temperatura:   float,
        humedad_aire:  float,
        cultivo:       int,
    ) -> float:
        """Probabilidad de necesitar riego (clase 1) para una zona."""
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a .train() primero.")
        X = self._row(humedad_suelo, temperatura, humedad_aire, cultivo)
        proba = self.model.predict_proba(X)[0]
        clases = list(self.model.classes_)
        if 1 in clases:
            return float(proba[clases.index(1)])
        # El modelo solo vio una clase durante el entrenamiento (dataset
        # filtrado muy desbalanceado). Devolvemos 1.0 si esa única clase
        # es "regar", o 0.0 si es "no regar", para que el resto de la
        # simulación siga funcionando sin romperse.
        return 1.0 if clases[0] == 1 else 0.0

    # ─── REPORTE ──────────────────────────────────────────────────────────────

    def print_evaluation(self) -> None:
        """Imprime un resumen legible de las métricas en consola."""
        if not self.metrics:
            print("  ⚠  Modelo no evaluado todavía.")
            return

        m = self.metrics
        print()
        print(f"  Fuente de datos: {m['fuente_datos']}")
        print(f"  Significado de 'regar': {m['significado_regar']}")
        if not m['evaluacion_valida']:
            print("  ⚠️  EVALUACIÓN NO VÁLIDA: el modelo se entrenó, pero su rendimiento "
                  "NO pudo evaluarse.")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │         EVALUACIÓN – ÁRBOL DE DECISIÓN              │")
        print("  ├─────────────────────────────────────────────────────┤")
        acc_txt = (f"{m['accuracy']:.4f}  ({m['accuracy']:.2%})"
                   if m['accuracy'] is not None else "no disponible")
        cv_txt  = (f"{m['cv_mean']:.4f} ± {m['cv_std']:.4f}"
                   if m['cv_mean'] is not None else "no disponible")
        print(f"  │  Accuracy (test):        {acc_txt:<26}│")
        print(f"  │  CV Accuracy:            {cv_txt:<26}│")
        print(f"  │  Muestras train/test:    {m['n_train']} / {m['n_test']}              │")
        print("  ├─────────────────────────────────────────────────────┤")
        if m['feature_importance'] is not None:
            print("  │  Importancia de variables:                          │")
            for feat, imp in m['feature_importance'].items():
                bar = '█' * round(imp * 25)
                print(f"  │    {feat:<20}  {imp:.4f}  {bar:<25}│")
        else:
            print("  │  Importancia de variables: no disponible            │")
        print("  └─────────────────────────────────────────────────────┘")
        for aviso in m['advertencias']:
            print(f"  ⚠️  {aviso}")
        if m['fuente_datos'].startswith("kaggle"):
            print("  ⚠️  Estas métricas miden qué tanto el árbol imita el historial de riego "
                  "de Kaggle, no la necesidad real de agua de maíz o arroz.")
        elif m['fuente_datos'].startswith("sintetico"):
            print("  ⚠️  Estas métricas miden qué tanto el árbol recupera la regla sintética "
                  "con que se etiquetó el dataset; no demuestran rendimiento en campo.")
        print()
        if m['classification_report'] is not None:
            print(m['classification_report'])
        else:
            print("  Informe de clasificación y matriz de confusión: no disponibles.")


# ════════════════════════════════════════════════════════════════════════════
# 4. SIMULACIÓN (7 días)
# ════════════════════════════════════════════════════════════════════════════
#
# Cada día:
#     1. Leer estado de las 8 zonas
#     2. Predecir riego con el árbol de decisión (por zona, incluye cultivo)
#     3. Aplicar restricción (máx. 4 zonas/día). Candidatas = zonas que el
#        árbol pide regar + zonas PRIORITARIAS por la regla de seguridad
#        (humedad < UMBRAL_HUMEDAD_SEGURIDAD). Orden: prioritarias primero,
#        de menor a mayor humedad (desempate: SCORE); luego las demás por
#        SCORE,
#        con desempates deterministas.
#     4. Activar válvulas (zonas seleccionadas)
#     5. Actualizar humedad (riego, lluvia o decaimiento), con reglas
#        DIFERENCIADAS por cultivo y lluvia limitada en zonas
#        ya húmedas

@dataclass
class RegistroDia:
    """Snapshot de un día de simulación, para análisis y gráficos."""
    dia: int
    humedad_inicio:  dict   # estado ANTES de actualizar
    humedad_final:   dict   # estado DESPUÉS de actualizar 
    regadas: List[str]
    llovio: bool
    litros_usados: int
    temperatura: float
    humedad_aire: float
    zonas_sanas_pct: float  # % de zonas con humedad_final >= UMBRAL_HUMEDAD_SANA
    zonas_prioritarias: List[str] = field(default_factory=list)
    prioritarias_sin_atender: List[str] = field(default_factory=list)


@dataclass
class ResultadoSimulacion:
    registros: List[RegistroDia] = field(default_factory=list)

    @property
    def litros_totales(self) -> int:
        return sum(r.litros_usados for r in self.registros)

    @property
    def ahorro_porcentual(self) -> float:
        return 1 - (self.litros_totales / CONSUMO_BASE_L)

    @property
    def consumo_maximo_posible(self) -> int:
        """Tope físico de la simulación: válvulas/día × L por válvula × días."""
        return len(self.registros) * MAX_ZONAS_POR_DIA * LITROS_POR_VALVULA

    @property
    def ahorro_minimo_garantizado(self) -> float:
        """
        Ahorro frente a CONSUMO_BASE_L que ya impone el límite de válvulas,
        sin importar la calidad de las decisiones (ni del modelo).
        """
        return max(0.0, 1 - self.consumo_maximo_posible / CONSUMO_BASE_L)

    def porcentaje_zonas_sanas_promedio(self) -> float:
        """% promedio (informativo) de zonas saludables a lo largo de la semana."""
        if not self.registros:
            return float("nan")   
        return float(np.mean([r.zonas_sanas_pct for r in self.registros]))

    def dias_que_fallan_el_umbral(self) -> List[int]:
        """Lista de días en que el % de zonas sanas cae bajo PORC_ZONAS_SANAS_MIN."""
        return [r.dia for r in self.registros if r.zonas_sanas_pct < PORC_ZONAS_SANAS_MIN]

    def dias_con_limite_de_valvulas(self) -> List[int]:
        """Días en que el límite de válvulas impidió regar todas las zonas prioritarias."""
        return [r.dia for r in self.registros if r.prioritarias_sin_atender]

    def semana_exitosa(self) -> dict:
        """
        Criterio de éxito ESTRICTO. Deben cumplirse TODOS:
            Litros <= CONSUMO_OBJETIVO_L
            Ahorro >= AHORRO_MINIMO respecto a CONSUMO_BASE_L
            EN TODOS LOS DÍAS, zonas sanas >= PORC_ZONAS_SANAS_MIN
        (antes se exigía solo el promedio semanal, lo que podía esconder
        días puntuales muy por debajo del umbral)

        Importante: los criterios de litros y ahorro son equivalentes entre
        sí con los valores actuales (420 L = 75% de 560 L) y, además, el
        límite de válvulas ya garantiza un consumo <= consumo_maximo_posible.
        Por eso 'ahorro_minimo_garantizado' y 'ahorro_adicional' se informan
        por separado. 'ahorro_adicional' es solo la diferencia aritmética entre
        el ahorro total y el garantizado (cuánto se consumió por debajo del
        tope de válvulas): NO mide la contribución del modelo. Para eso solo
        sirve la comparación con la estrategia de referencia sin ML.
        Si no hay días simulados devuelve 'evaluable' = False y 'exito' = False.
        """
        if not self.registros:
            return {"evaluable": False, "exito": False, "litros_totales": 0,
                    "dias_fallidos": [], "dias_limite_valvulas": []}

        ahorro_ok = self.ahorro_porcentual >= AHORRO_MINIMO
        litros_ok = self.litros_totales <= CONSUMO_OBJETIVO_L
        dias_fallidos = self.dias_que_fallan_el_umbral()
        zonas_ok = len(dias_fallidos) == 0

        return {
            "evaluable":             True,
            "litros_totales":        self.litros_totales,
            "consumo_base":          CONSUMO_BASE_L,
            "consumo_maximo_posible": self.consumo_maximo_posible,
            "ahorro_porcentual":     self.ahorro_porcentual,
            "ahorro_cumple":         ahorro_ok,
            "ahorro_minimo_garantizado": self.ahorro_minimo_garantizado,
            "ahorro_adicional":      self.ahorro_porcentual - self.ahorro_minimo_garantizado,
            "zonas_sanas_pct_prom":  self.porcentaje_zonas_sanas_promedio(),
            "dias_fallidos":         dias_fallidos,
            "zonas_sanas_cumple":    zonas_ok,
            "dias_limite_valvulas":  self.dias_con_limite_de_valvulas(),
            "objetivo_litros":       CONSUMO_OBJETIVO_L,
            "litros_cumple":         litros_ok,
            "exito":                 ahorro_ok and litros_ok and zonas_ok,
        }


def _clima_del_dia(rng: random.Random) -> tuple:
    """Genera temperatura, humedad de aire, y si llueve (5% de prob.)."""
    temperatura  = round(rng.normalvariate(22, 5), 1)
    humedad_aire = round(rng.uniform(30, 90), 1)
    llovio       = rng.random() < PROB_LLUVIA
    return temperatura, humedad_aire, llovio


def _aplicar_lluvia(humedad: float) -> float:
    if humedad >= UMBRAL_SATURACION_LLUVIA:
        return humedad
    return float(np.clip(humedad + INCREMENTO_LLUVIA, HUMEDAD_MIN, HUMEDAD_MAX))


def _aplicar_decaimiento(humedad: float, cultivo: str, llovio: bool) -> float:
    """Zona NO regada: decae según su cultivo, y además recibe lluvia si llovió."""
    decaimiento = REGLAS_POR_CULTIVO[cultivo]["decaimiento_sin_riego"]
    nueva = humedad - decaimiento
    nueva = float(np.clip(nueva, HUMEDAD_MIN, HUMEDAD_MAX))
    if llovio:
        nueva = _aplicar_lluvia(nueva)
    return nueva


def _aplicar_riego(humedad: float, cultivo: str, llovio: bool) -> float:
    """Zona SÍ regada: sube según su cultivo, y además recibe lluvia si llovió."""
    incremento = REGLAS_POR_CULTIVO[cultivo]["incremento_con_riego"]
    nueva = humedad + incremento
    nueva = float(np.clip(nueva, HUMEDAD_MIN, HUMEDAD_MAX))
    if llovio:
        nueva = _aplicar_lluvia(nueva)
    return nueva


def _cultivo_de_zona(zid: str) -> str:
    return next(z["cultivo"] for z in ZONAS if z["id"] == zid)


_ORDEN_ZONAS = {z["id"]: i for i, z in enumerate(ZONAS)}


def _seleccionar_zonas(candidatas: list, max_zonas: int = MAX_ZONAS_POR_DIA) -> tuple:
    """
    Elige qué zonas regar entre las candidatas, de forma determinista.

    candidatas: lista de (zid, humedad, probabilidad, score, prioritaria).
    Orden de atención:
        1. Zonas prioritarias (humedad bajo UMBRAL_HUMEDAD_SEGURIDAD) antes
           que las demás.
        2. Entre las prioritarias: la de MENOR humedad primero. Si hay empate
           de humedad, mayor score.
        3. Entre las no prioritarias: mayor score primero, si hay empate,
           menor humedad.
        4. Último desempate en ambos grupos: orden de ZONAS.
    Devuelve (zonas_a_regar, prioritarias_sin_atender). Nunca devuelve más de
    max_zonas zonas; 'prioritarias_sin_atender' lista las prioritarias que
    quedaron fuera porque el límite de válvulas lo impidió.
    """
    def clave(t):
        zid, humedad, _prob, score, prioritaria = t
        orden = _ORDEN_ZONAS.get(zid, len(_ORDEN_ZONAS))
        if prioritaria:
            return (0, humedad, -score, orden, zid)
        return (1, -score, humedad, orden, zid)

    ordenadas = sorted(candidatas, key=clave)
    zonas_a_regar = [t[0] for t in ordenadas[:max_zonas]]
    sin_atender = [t[0] for t in ordenadas[max_zonas:] if t[4]]
    return zonas_a_regar, sin_atender


def ejecutar_simulacion(
    modelo: Optional[IrrigationModel],
    dias: int = DIAS_SIMULACION,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
    usar_modelo: bool = True,
) -> ResultadoSimulacion:
    """
    Corre la simulación completa día por día y devuelve un
    ResultadoSimulacion con el historial para evaluación y gráficos.
    """
    if dias < 1:
        raise ValueError(f"'dias' debe ser >= 1 (recibido: {dias}).")
    if usar_modelo and modelo is None:
        raise ValueError("Se necesita un modelo entrenado (o usar_modelo=False).")
    rng = random.Random(seed)
    humedad_actual = dict(HUMEDAD_INICIAL)
    resultado = ResultadoSimulacion()

    for dia in range(1, dias + 1):
        temperatura, humedad_aire, llovio = _clima_del_dia(rng)
        humedad_inicio_dia = dict(humedad_actual)

        candidatas = []   # (zid, humedad, probabilidad, score, prioritaria)
        for zona in ZONAS:
            zid     = zona["id"]
            cultivo = CODIGO_CULTIVO[zona["cultivo"]]
            humedad = humedad_actual[zid]

            if usar_modelo:
                necesita_riego = modelo.predict_zone(humedad, temperatura, humedad_aire, cultivo)
                prob = modelo.predict_proba_zone(humedad, temperatura, humedad_aire, cultivo)
            else:
                necesita_riego, prob = 0, 0.0
            criticidad = (100 - humedad) / 100
            score = PESO_PROBABILIDAD * prob + (1 - PESO_PROBABILIDAD) * criticidad
            prioritaria = humedad < UMBRAL_HUMEDAD_SEGURIDAD
            if necesita_riego == 1 or prioritaria:
                candidatas.append((zid, humedad, prob, score, prioritaria))
        zonas_a_regar, prioritarias_sin_atender = _seleccionar_zonas(candidatas)
        zonas_prioritarias = [c[0] for c in candidatas if c[4]]

        nueva_humedad = {}
        for zona in ZONAS:
            zid     = zona["id"]
            cultivo = zona["cultivo"]
            h_actual = humedad_actual[zid]
            if zid in zonas_a_regar:
                nueva_humedad[zid] = _aplicar_riego(h_actual, cultivo, llovio)
            else:
                nueva_humedad[zid] = _aplicar_decaimiento(h_actual, cultivo, llovio)

        litros_usados = len(zonas_a_regar) * LITROS_POR_VALVULA

        # % de zonas sanas se calcula sobre el estado FINAL del día (post-riego)
        sanas = sum(1 for h in nueva_humedad.values() if h >= UMBRAL_HUMEDAD_SANA)
        zonas_sanas_pct = sanas / len(ZONAS)

        registro = RegistroDia(
            dia=dia,
            humedad_inicio=humedad_inicio_dia,
            humedad_final=dict(nueva_humedad),   
            regadas=zonas_a_regar,
            llovio=llovio,
            litros_usados=litros_usados,
            temperatura=temperatura,
            humedad_aire=humedad_aire,
            zonas_sanas_pct=zonas_sanas_pct,
            zonas_prioritarias=zonas_prioritarias,
            prioritarias_sin_atender=prioritarias_sin_atender,
        )
        resultado.registros.append(registro)

        if verbose:
            _imprimir_dia(registro)

        humedad_actual = nueva_humedad

    return resultado


def _imprimir_dia(registro: RegistroDia) -> None:
    lluvia_txt = "🌧️  Sí" if registro.llovio else "☀️  No"
    print(f"\n  ── Día {registro.dia} " + "─" * 40)
    print(f"  Temperatura: {registro.temperatura:.1f}°C   "
          f"Humedad aire: {registro.humedad_aire:.1f}%   Lluvia: {lluvia_txt}")
    print(f"  Zonas regadas ({len(registro.regadas)}/{MAX_ZONAS_POR_DIA}): "
          f"{', '.join(registro.regadas) if registro.regadas else '— ninguna —'}")
    if registro.prioritarias_sin_atender:
        print(f"  ⚠️  Límite de {MAX_ZONAS_POR_DIA} válvulas: no se pudo atender a las zonas "
              f"prioritarias {', '.join(registro.prioritarias_sin_atender)}")
    print(f"  Agua usada hoy: {registro.litros_usados} L   "
          f"Zonas saludables (fin del día): {registro.zonas_sanas_pct:.0%}")
    print(f"  {'Zona':<6}{'Cultivo':<10}{'Humedad inicio':>16}{'Humedad final':>16}{'Regada':>10}")
    for zona in ZONAS:
        zid = zona["id"]
        marca = "✅" if zid in registro.regadas else "—"
        print(f"  {zid:<6}{zona['cultivo']:<10}{registro.humedad_inicio[zid]:>15.1f}%"
              f"{registro.humedad_final[zid]:>15.1f}%{marca:>10}")


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIONES
# ════════════════════════════════════════════════════════════════════════════

def _finalizar_figura(fig, guardar_como: Optional[str] = None) -> None:
    """Guarda (creando la carpeta si hace falta), muestra y cierra la figura."""
    if guardar_como:
        Path(guardar_como).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(guardar_como, dpi=150, bbox_inches='tight')
        print(f"  📈 Gráfico guardado en '{guardar_como}'")
    plt.show()
    plt.close(fig)


def _sin_registros(resultado: ResultadoSimulacion) -> bool:
    if not resultado.registros:
        print("  ⚠️  Sin días simulados: no hay nada que graficar.")
        return True
    return False


def graficar_humedad_por_zona(resultado: ResultadoSimulacion, guardar_como: str = None) -> None:
    """
    Línea de tiempo de humedad para cada una de las 8 zonas.
    Usa humedad_final (estado real post-actualización) — Problema 5 corregido.
    Incluye el punto del día 0 (humedad inicial) para que la curva sea completa.
    """
    if _sin_registros(resultado):
        return
    dias = [0] + [r.dia for r in resultado.registros]
    zona_ids = [z["id"] for z in ZONAS]

    fig, ax = plt.subplots(figsize=(10, 6))

    for zid in zona_ids:
        valores = [HUMEDAD_INICIAL[zid]] + [r.humedad_final[zid] for r in resultado.registros]
        ax.plot(dias, valores, marker='o', label=zid)

    ax.axhline(UMBRAL_HUMEDAD_SANA, color='red', linestyle='--', linewidth=1,
               label=f'Umbral saludable ({UMBRAL_HUMEDAD_SANA}%)')

    ax.set_title('Evolución de humedad del suelo por zona (post-riego/decaimiento diario)')
    ax.set_xlabel('Día')
    ax.set_ylabel('Humedad del suelo (%)')
    ax.set_xticks(dias)
    ax.legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _finalizar_figura(fig, guardar_como)


def graficar_consumo_diario(resultado: ResultadoSimulacion, guardar_como: str = None) -> None:
    """Barras de litros usados cada día, con la línea de promedio objetivo."""
    if _sin_registros(resultado):
        return
    dias = [r.dia for r in resultado.registros]
    litros = [r.litros_usados for r in resultado.registros]
    promedio_objetivo_diario = CONSUMO_OBJETIVO_L / len(dias)

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(dias, litros, color='#3a7d44', alpha=0.85, label='Agua usada')

    for barra, valor in zip(barras, litros):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 1,
                 f'{valor}L', ha='center', fontsize=9)

    ax.axhline(promedio_objetivo_diario, color='orange', linestyle='--',
               label=f'Promedio objetivo (~{promedio_objetivo_diario:.0f}L/día)')

    ax.set_title('Consumo de agua diario durante la simulación')
    ax.set_xlabel('Día')
    ax.set_ylabel('Litros usados')
    ax.set_xticks(dias)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()

    _finalizar_figura(fig, guardar_como)


def graficar_comparacion_consumo(resultado: ResultadoSimulacion, guardar_como: str = None) -> None:
    """Barras comparando Consumo Base vs. Objetivo vs. Real (total semanal)."""
    if _sin_registros(resultado):
        return
    categorias = ['Base\n(sin optimizar)', 'Objetivo\n(meta)', 'Real\n(simulado)']
    valores = [CONSUMO_BASE_L, CONSUMO_OBJETIVO_L, resultado.litros_totales]
    colores = ['#b0413e', '#e8a33d', '#3a7d44']

    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(categorias, valores, color=colores, alpha=0.9)

    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 8,
                 f'{valor:.0f}L', ha='center', fontsize=10, fontweight='bold')

    ax.set_title(f'Consumo total semanal — Ahorro vs. base: {resultado.ahorro_porcentual:.1%}\n'
                 f'(el límite de válvulas ya garantiza ≥ {resultado.ahorro_minimo_garantizado:.0%})',
                 fontsize=11)
    ax.set_ylabel('Litros')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()

    _finalizar_figura(fig, guardar_como)


def graficar_zonas_sanas_por_dia(resultado: ResultadoSimulacion, guardar_como: str = None) -> None:
    """
    Muestra el % de zonas saludables AL FINAL de cada día, comparado contra el umbral
    de 80% que se exige TODOS los días, no en promedio.
    """
    if _sin_registros(resultado):
        return
    dias = [r.dia for r in resultado.registros]
    porcentajes = [r.zonas_sanas_pct * 100 for r in resultado.registros]
    colores = ['#3a7d44' if p >= PORC_ZONAS_SANAS_MIN * 100 else '#b0413e' for p in porcentajes]

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(dias, porcentajes, color=colores, alpha=0.9)

    for barra, valor in zip(barras, porcentajes):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 1.5,
                 f'{valor:.0f}%', ha='center', fontsize=9)

    ax.axhline(PORC_ZONAS_SANAS_MIN * 100, color='black', linestyle='--',
               label=f'Umbral exigido ({PORC_ZONAS_SANAS_MIN:.0%}, todos los días)')

    ax.set_title('% de zonas saludables al final de cada día')
    ax.set_xlabel('Día')
    ax.set_ylabel('% zonas con humedad >= 40%')
    ax.set_xticks(dias)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()

    _finalizar_figura(fig, guardar_como)


def graficar_matriz_confusion(metrics: dict, guardar_como: str = None) -> None:
    """Heatmap simple de la matriz de confusión del modelo."""
    cm = metrics.get('confusion_matrix')
    if cm is None:
        print("  ⚠️  Matriz de confusión no disponible (el modelo no pudo evaluarse): "
              "no se genera el gráfico.")
        return
    labels = ['No Regar', 'Regar']

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Greens')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f"Matriz de Confusión (Accuracy: {metrics['accuracy']:.2%})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=14, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    _finalizar_figura(fig, guardar_como)


def generar_todos_los_graficos(resultado: ResultadoSimulacion, metrics: dict,
                                carpeta_salida: str = None) -> None:
    """Conveniencia: genera y muestra los 5 gráficos en secuencia (crea la carpeta de salida)."""
    if carpeta_salida:
        Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    def ruta(nombre):
        return str(Path(carpeta_salida) / nombre) if carpeta_salida else None

    graficar_matriz_confusion(metrics, ruta("matriz_confusion.png"))
    graficar_humedad_por_zona(resultado, ruta("humedad_por_zona.png"))
    graficar_consumo_diario(resultado, ruta("consumo_diario.png"))
    graficar_comparacion_consumo(resultado, ruta("comparacion_consumo.png"))
    graficar_zonas_sanas_por_dia(resultado, ruta("zonas_sanas_por_dia.png"))


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(modo_datos: str = MODO_DATOS, carpeta_salida: Optional[str] = None) -> None:
    print("=" * 60)
    print(" AGRICULTURA INTELIGENTE ")
    print("=" * 60)

    # 1. Dataset
    print("\n[1/4] Preparando dataset...")
    df = load_dataset(modo=modo_datos)

    # 2. Entrenamiento y evaluación del modelo
    print("\n[2/4] Entrenando árbol de decisión...")
    modelo = IrrigationModel()
    metrics = modelo.train(df)
    modelo.print_evaluation()

    # 3. Simulación de 7 días
    print(f"\n[3/4] Simulando {DIAS_SIMULACION} días de riego sobre 8 zonas...")
    resultado = ejecutar_simulacion(modelo, dias=DIAS_SIMULACION, verbose=True)

    # 4. Evaluación del criterio de éxito final 
    print("\n[4/4] Evaluando criterio de éxito de la semana...")
    evaluacion = resultado.semana_exitosa()
    _imprimir_resumen_final(evaluacion, metrics["fuente_datos"])

    # Estrategia de referencia SIN Machine Learning, mismas condiciones
    referencia = ejecutar_simulacion(None, dias=DIAS_SIMULACION, verbose=False, usar_modelo=False)
    _imprimir_comparacion(resultado, referencia)

    # 5. Gráficos
    print("\nGenerando gráficos...")
    generar_todos_los_graficos(resultado, metrics, carpeta_salida)


def _imprimir_resumen_final(ev: dict, fuente_datos: Optional[str] = None) -> None:
    if not ev.get("evaluable", True):
        print("\n  ⚠️  No hay días simulados: no se puede evaluar la semana.")
        return
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │              RESUMEN DE LA SEMANA                      │")
    print("  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Litros totales usados:     {ev['litros_totales']:.0f} L"
          f"{'':<10}│")
    print(f"  │  Objetivo (<= {ev['objetivo_litros']} L):       "
          f"{'✅ Cumple' if ev['litros_cumple'] else '❌ No cumple'}"
          f"{'':<10}│")
    print(f"  │  Ahorro vs. base ({ev['consumo_base']}L):    {ev['ahorro_porcentual']:.1%}  "
          f"({'✅ >= ' if ev['ahorro_cumple'] else '❌ < '}{AHORRO_MINIMO:.0%}){'':<5}│")
    print(f"  │  Zonas sanas (promedio):    {ev['zonas_sanas_pct_prom']:.1%}"
          f"{'':<14}│")
    if ev["dias_fallidos"]:
        dias_str = ", ".join(str(d) for d in ev["dias_fallidos"])
        print(f"  │  Días que NO llegan a {PORC_ZONAS_SANAS_MIN:.0%}:  {dias_str:<27}│")
        print(f"  │  Criterio estricto >={PORC_ZONAS_SANAS_MIN:.0%}/día: ❌ No cumple{'':<13}│")
    else:
        print(f"  │  Criterio estricto >={PORC_ZONAS_SANAS_MIN:.0%}/día: ✅ Cumple TODOS los días{'':<0}│")
    print("  ├─────────────────────────────────────────────────────────┤")
    resultado_final = "🏆 SEMANA EXITOSA" if ev["exito"] else "⚠️  SEMANA NO EXITOSA"
    print(f"  │  {resultado_final:<55}│")
    print("  └─────────────────────────────────────────────────────────┘")

    # Cómo interpretar el consumo y el ahorro 
    base_teorica = len(ZONAS) * LITROS_POR_VALVULA * DIAS_SIMULACION
    print("  Cómo leer el consumo y el ahorro:")
    print(f"   • Consumo base ({ev['consumo_base']} L): referencia configurada"
          + (f" (= regar las {len(ZONAS)} zonas todos los días: "
             f"{len(ZONAS)} × {LITROS_POR_VALVULA} L × {DIAS_SIMULACION})."
             if base_teorica == ev['consumo_base'] else "."))
    print(f"   • El límite de {MAX_ZONAS_POR_DIA} válvulas/día impone un máximo de "
          f"{ev['consumo_maximo_posible']} L: ahorro MÍNIMO GARANTIZADO de "
          f"{ev['ahorro_minimo_garantizado']:.1%} sin importar el modelo.")
    print(f"   • Consumo real: {ev['litros_totales']:.0f} L de un máximo posible de "
          f"{ev['consumo_maximo_posible']} L ({ev['litros_totales'] / ev['consumo_maximo_posible']:.0%} "
          f"del tope de válvulas).")
    print(f"   • La diferencia entre el ahorro total y el garantizado "
          f"({ev['ahorro_adicional'] * 100:.1f} puntos porcentuales) solo indica cuánto se "
          f"consumió por debajo del tope de válvulas; NO mide la aportación del Machine Learning "
          f"(ver la comparación con la estrategia de referencia).")
    if ev['consumo_maximo_posible'] <= ev['objetivo_litros']:
        print(f"   • Con estos parámetros los criterios de litros y de ahorro se cumplen "
              f"por construcción; el único criterio que discrimina es el de zonas sanas.")
    if ev["dias_limite_valvulas"]:
        print(f"   • El límite de válvulas impidió atender todas las zonas prioritarias "
              f"los días: {', '.join(str(d) for d in ev['dias_limite_valvulas'])}.")
    if fuente_datos:
        print(f"   • Datos de entrenamiento: {fuente_datos} "
              f"({'indirectos, experimental' if fuente_datos.startswith('kaggle') else 'sintéticos, no son datos reales'}).")


def _mismas_condiciones(res_a: ResultadoSimulacion, res_b: ResultadoSimulacion) -> bool:
    """True si ambas simulaciones tuvieron el mismo inicio, duración y clima diario."""
    if len(res_a.registros) != len(res_b.registros):
        return False
    if res_a.registros[0].humedad_inicio != res_b.registros[0].humedad_inicio:
        return False
    return all((a.temperatura, a.humedad_aire, a.llovio) == (b.temperatura, b.humedad_aire, b.llovio)
               for a, b in zip(res_a.registros, res_b.registros))


def _imprimir_comparacion(res_ml: ResultadoSimulacion, res_ref: ResultadoSimulacion) -> None:
    """
    Compara con la estrategia de referencia SIN ML bajo las mismas condiciones.

    La referencia aplica la misma regla de seguridad y el mismo límite de
    válvulas, así que la diferencia refleja solo lo que el árbol añade a esa
    regla en ESTA semana simulada. No es una medida general del modelo.
    """
    if not res_ml.registros or not res_ref.registros:
        print("\n  ⚠️  Sin días simulados en alguna de las estrategias: no hay comparación posible.")
        return
    if not _mismas_condiciones(res_ml, res_ref):
        print("\n  ⚠️  Las dos simulaciones no tuvieron las mismas condiciones (duración, humedades "
              "iniciales o clima): la comparación no es válida y no se muestra.")
        return

    filas = [("Árbol + regla de seguridad", res_ml), ("Solo regla de seguridad (sin ML)", res_ref)]
    print()
    print("  Comparación con estrategia de referencia (misma semilla, clima y válvulas):")
    print(f"  {'Estrategia':<34}{'Litros':>8}{'Zonas sanas':>13}{'Días <'+format(PORC_ZONAS_SANAS_MIN, '.0%'):>10}{'Éxito':>8}")
    for nombre, res in filas:
        ev = res.semana_exitosa()
        print(f"  {nombre:<34}{res.litros_totales:>8}{ev['zonas_sanas_pct_prom']:>13.1%}"
              f"{len(ev['dias_fallidos']):>10}{'sí' if ev['exito'] else 'no':>8}")

    d_litros = res_ml.litros_totales - res_ref.litros_totales
    d_sanas  = (res_ml.porcentaje_zonas_sanas_promedio()
                - res_ref.porcentaje_zonas_sanas_promedio()) * 100
    print(f"  {'Diferencia (ML − referencia)':<34}{d_litros:>+8}{d_sanas:>+12.1f} pp")

    if d_litros > 0:
        agua_txt = f"usó {d_litros} L más de agua"
        if res_ref.litros_totales > 0:
            agua_txt += f" ({d_litros / res_ref.litros_totales:.1%} más)"
    elif d_litros < 0:
        agua_txt = f"usó {-d_litros} L menos de agua"
        if res_ref.litros_totales > 0:
            agua_txt += f" ({-d_litros / res_ref.litros_totales:.1%} menos)"
    else:
        agua_txt = "usó la misma cantidad de agua"
    if abs(d_sanas) < 1e-9:
        humedad_txt = "conservó igual la humedad (mismo % promedio de zonas sanas)"
    elif d_sanas > 0:
        humedad_txt = f"conservó mejor la humedad ({d_sanas:+.1f} puntos porcentuales de zonas sanas)"
    else:
        humedad_txt = f"conservó peor la humedad ({d_sanas:+.1f} puntos porcentuales de zonas sanas)"
    print(f"  En esta semana simulada, la estrategia con ML {agua_txt} que la referencia y {humedad_txt}.")
    print("  Ambas comparten el límite de válvulas, por lo que el ahorro que éste garantiza no se "
          "atribuye al ML.")
    print("  Nota: es UNA sola semana simulada con una semilla; no demuestra superioridad general "
          "del modelo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agricultura Inteligente")
    parser.add_argument("--datos", choices=MODOS_DATOS, default=MODO_DATOS,
                        help="origen de los datos: auto (por defecto), sintetico o kaggle")
    parser.add_argument("--salida", default=None,
                        help="carpeta donde guardar los gráficos (se crea si no existe)")
    args = parser.parse_args()
    try:
        main(modo_datos=args.datos, carpeta_salida=args.salida)
    except ValueError as error:
        print(f"\n  ❌ Error: {error}", file=sys.stderr)
        sys.exit(1)
