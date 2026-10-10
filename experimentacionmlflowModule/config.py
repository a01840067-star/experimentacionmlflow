from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
UCI_DATASET_ID = 9
RAW_DATA_FILE = RAW_DATA_DIR / "auto_mpg.csv"
RANDOM_STATE= 42
TARGET = "mpg"
ID_COL = "car_name"
ORIGIN_MAP = {1: "USA", 2: "Europa", 3: "Japón"}