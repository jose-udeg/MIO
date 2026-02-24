# config.py
# Parámetros globales del proyecto — cambia lo que necesites.

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Rutas de datos
DATA_RAW = os.path.join(BASE_DIR, "data", "raw") if os.path.exists(os.path.join(BASE_DIR, "data")) else os.path.join(BASE_DIR, "..", "data", "raw")
NEWS_CSV = os.path.join(DATA_RAW, "news.csv")
PRICES_CSV = os.path.join(DATA_RAW, "prices.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Time / ventanas
TIME_COL = "timestamp"
TIMEFRAME = "1H"            # resolución esperada de prices.csv
WINDOW_HOURS = 24           # horizonte lookahead
NUMERIC_WINDOW = 24         # T pasos previos para la rama numérica

# Etiquetado
THRESHOLD = 0.003           # umbral para ternario (ajustar por activo/timeframe)

# Texto
MAX_WORDS = 20000
MAX_LEN = 100

# Entrenamiento
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
RANDOM_SEED = 42

# Model output
MODEL_DIR = os.path.join(BASE_DIR, "..", "models_checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)
