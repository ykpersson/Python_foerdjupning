"""
Ren och modern konfiguration för projektets katalogstruktur, datumintervall och tickerlista.
Denna version eliminerar oklarheter, kompatibilitetshack och dolda path-fel.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Baspath för hela projektet
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Datamappar (konsekvent struktur)
# ---------------------------------------------------------
DATA_DIR     = BASE / "data"
PRICE_DIR    = DATA_DIR / "price"
TECH_DIR     = DATA_DIR / "technical"
TARGET_DIR   = DATA_DIR / "target"
FEATURE_DIR  = DATA_DIR / "features"

# ---------------------------------------------------------
# Dataset (EN källa till sanning)
# ---------------------------------------------------------
DATASET_PATH = DATA_DIR / "dataset_all.pkl"

# ---------------------------------------------------------
# Splits och testresultat
# ---------------------------------------------------------
SPLIT_DIR = BASE / "models" / "splits"
TEST_DIR  = BASE / "models" / "test_results"

# ---------------------------------------------------------
# Loggar
# ---------------------------------------------------------
LOG_DIR        = BASE / "logs"
PRICE_LOG_DIR  = PRICE_DIR / "price_log"
TECH_LOG_DIR   = TECH_DIR / "tech_log"
TARGET_LOG_DIR = TARGET_DIR / "target_log"

# ---------------------------------------------------------
# Datumintervall (fixerat, inte dynamiskt)
# ---------------------------------------------------------
YEARS = 5

# ---------------------------------------------------------
# Tickerlista
# ---------------------------------------------------------
tickers = [
    "ABB.ST", "ALFA.ST", "ALIV-SDB.ST", "ATCO-A.ST", "ATCO-B.ST",
    "AZN.ST", "BOL.ST", "ELUX-B.ST", "EVO.ST", "ERIC-B.ST",
    "ESSITY-B.ST", "GETI-B.ST", "HEXA-B.ST", "HM-B.ST", "HOLM-B.ST",
    "INVE-B.ST", "NDA-SE.ST", "SAND.ST", "SCA-B.ST", "SEB-A.ST",
    "SHB-A.ST", "SINCH.ST", "SKF-B.ST", "SWED-A.ST", "TELIA.ST",
    "VOLV-B.ST", "EQNR.ST", "TEL2-B.ST", "SKA-B.ST", "EQT.ST",
]

# ---------------------------------------------------------
# Filstrukturen (PKL-version)
# ---------------------------------------------------------
PRICE_FILES  = [PRICE_DIR  / f"price_{t}.pkl"  for t in tickers]
TECH_FILES   = [TECH_DIR   / f"tech_{t}.pkl"   for t in tickers]
TARGET_FILES = [TARGET_DIR / f"target_{t}.pkl" for t in tickers]
