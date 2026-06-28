import os
import joblib
import setting
import pandas as pd


"""
Generella hjälpfunktioner för filhantering i projektet.
Innehåller save/load-funktioner, katalogskapande och filnamnssanering.
"""


def get_date_range():
    """
    Returnerar tre datum:
    - tech_start: för Price/Tech (sliding windows)
    - start: för Target/Features/Dataset/Splits
    - end: slutdatum (idag)
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=5)
    tech_start = start - pd.DateOffset(days=252)
    return tech_start, start, end

def normalize_ticker(t):
    return str(t).upper().strip()


# ---------------------------------------------------------
# Filnamnssanering
# ---------------------------------------------------------

def _safe(name: str) -> str:
    """
    Gör en sträng filsystems-säker genom att ersätta tecken
    som inte funkar i filnamn.
    """
    return name.replace(".", "_").replace("-", "_").strip()


# ---------------------------------------------------------
# Kataloghantering
# ---------------------------------------------------------

def get_all_dirs() -> list[str]:
    """
    Returnerar alla kataloger som ska skapas baserat på setting.py.
    Dynamisk för att fungera med monkeypatch/reload i tester.
    """
    return [
        setting.DATA_DIR,
        setting.PRICE_DIR,
        setting.TECH_DIR,
        setting.TARGET_DIR,
        setting.DATASET_DIR,
        setting.TEST_DIR,
        setting.LOG_DIR,
        setting.PRICE_LOG_DIR,
        setting.TECH_LOG_DIR,
        setting.TARGET_LOG_DIR,
    ]


def build_directories() -> None:
    """
    Skapar alla kataloger som definieras i get_all_dirs().
    """
    for d in get_all_dirs():
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------
# Save / Load wrappers (joblib)
# ---------------------------------------------------------

def ensure_parent_dir(path: str) -> None:
    """
    Säkerställer att föräldrakatalogen till en fil finns.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pkl(obj, path: str) -> None:
    """
    Sparar ett objekt till en PKL-fil med joblib.dump.
    Skapar automatiskt katalogen om den saknas.
    """
    ensure_parent_dir(path)
    joblib.dump(obj, path)


def load_pkl(path: str):
    """
    Laddar ett objekt från en PKL-fil med joblib.load.
    Returnerar None om filen inte finns.
    """
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------
# Master save/load (projektets officiella format)
# ---------------------------------------------------------

def save_master(df, name: str) -> None:
    """
    Sparar en DataFrame som masterfil i data/-katalogen.
    Filnamnet saneras automatiskt.
    """
    filename = _safe(name) + ".pkl"
    path = os.path.join(setting.DATA_DIR, filename)
    ensure_parent_dir(path)
    joblib.dump(df, path)
    print(f"✓ Sparat master: {path}")


def load_master(name: str):
    """
    Laddar en masterfil från data/-katalogen.
    Returnerar None om filen saknas.
    """
    filename = _safe(name) + ".pkl"
    path = os.path.join(setting.DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------
# Små hjälpfunktioner
# ---------------------------------------------------------

def file_exists(path: str) -> bool:
    """
    Returnerar True om filen existerar.
    """
    return os.path.exists(path)

