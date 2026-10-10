import pandas as pd
from ucimlrepo import fetch_ucirepo
from experimentacionmlflowModule.config import UCI_DATASET_ID, RAW_DATA_FILE, RAW_DATA_DIR

def cargar_datos():
    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE)
    try:
        auto_mpg = fetch_ucirepo(id=UCI_DATASET_ID)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo descargar el dataset {UCI_DATASET_ID} de UCI. "
            "Revisa tu conexion o coloca el CSV manualmente en "
            f"{RAW_DATA_FILE}"
        ) from e

    df = auto_mpg.data.original.copy()

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
    except OSError as e:
            print(f"Aviso: no se pudo guardar la copia local ({e}). Se continua sin cache.")

    return df
