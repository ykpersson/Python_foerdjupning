import os

# === Projektets root ===
BASE = os.path.dirname(os.path.dirname(__file__))

# === DATASET ===
# Hela datasetet som används av train/test/evaluate
DATASET_PATH = "../data/dataset_all.pkl"

# === MODELLMAPPA ===
# Här sparas slutmodellerna
MODELS_FINAL_DIR = os.path.join(BASE, "models_final")

# Här sparas optimerade modeller (HPO)
MODELS_OPTIMIZED_DIR = os.path.join(BASE, "models_optimized")

# Skapa mappar om de inte finns
os.makedirs(MODELS_FINAL_DIR, exist_ok=True)
os.makedirs(MODELS_OPTIMIZED_DIR, exist_ok=True)

# === TICKERS ===
Tickers = [
    "ABB.ST", "ALFA.ST", "ALIV-SDB.ST", "ATCO-A.ST", "ATCO-B.ST",
    "AZN.ST", "BOL.ST", "ELUX-B.ST", "EVO.ST", "ERIC-B.ST",
    "ESSITY-B.ST", "GETI-B.ST", "HEXA-B.ST", "HM-B.ST", "HOLM-B.ST",
    "INVE-B.ST", "NDA-SE.ST", "SAND.ST", "SCA-B.ST", "SEB-A.ST",
    "SHB-A.ST", "SINCH.ST", "SKF-B.ST", "SWED-A.ST", "TELIA.ST",
    "VOLV-B.ST", "EQNR.ST", "TEL2-B.ST", "SKA-B.ST", "EQT.ST",
]

# === GRUPPER ===
GROUPS = ["G1", "G2", "G3", "G4"]