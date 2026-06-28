import pandas as pd
import numpy as np
import logger
from support_functions import load_master, save_master, get_date_range


def band_calc(df, band_name, col="rel_move"):
    """
    Minimal bandbaserad target:
    -1, 0, 1 → sedan mappas till 0,1,2
    """

    x = df[col]

    # Percentiler per band (ABS MINIMUM)
    if band_name == "G1":      # Mean reversion
        low_p  = x.quantile(0.20)
        high_p = x.quantile(0.80)

    elif band_name == "G2":    # Svag mean reversion
        low_p  = x.quantile(0.30)
        high_p = x.quantile(0.70)

    elif band_name == "G3":    # Mild momentum
        low_p  = x.quantile(0.40)
        high_p = x.quantile(0.60)

    elif band_name == "G4":    # Stark momentum
        low_p  = x.quantile(0.45)
        high_p = x.quantile(0.55)

    else:
        raise ValueError(f"Okänt band: {band_name}")

    # Target-logik
    df["target"] = np.where(
        x <= low_p, -1,
        np.where(x >= high_p, 1, 0)
    )

    return df

class TargetBuilder:

    def __init__(self):
        self.tech_start, self.start, self.end = get_date_range()
        self.bands = ["G1", "G2", "G3", "G4"]

        # Ladda PRICE + INDEX
        self.price = load_master("price_ohlc_raw")
        self.index = load_master("index_all")

        # Datumklipp
        self.price = self.price[self.price["Date"] >= self.tech_start].copy()
        self.index = self.index[self.index["Date"] >= self.tech_start].copy()

        if "Index_h" not in self.index.columns:
            raise ValueError("Index_all saknar kolumnen 'Index_h'!")

    def merge_price_index(self):
        return self.price.merge(self.index, on="Date", how="left")

    def compute_rel_move(self, df):
        df["rel_move"] = df["Return_h"] - df["Index_h"]
        return df

    def ordinal_target(self, df):
        df["y"] = pd.qcut(df["rel_move"], q=3, labels=[-1, 0, 1])
        return df

    def build_all(self):
        logger.log("Startar TargetBuilder...")

        base = self.merge_price_index()

        # Bandlistan (från BandBuilder-loggen)
        bands = {
            "G1": ['TELIA.ST', 'INVE-B.ST', 'ESSITY-B.ST', 'SHB-A.ST', 'NDA-SE.ST', 'TEL2-B.ST', 'SWED-A.ST'],
            "G2": ['HOLM-B.ST', 'AZN.ST', 'SEB-A.ST', 'ABB.ST', 'SCA-B.ST', 'ALFA.ST', 'VOLV-B.ST', 'SKA-B.ST'],
            "G3": ['SAND.ST', 'ERIC-B.ST', 'ATCO-B.ST', 'ATCO-A.ST', 'ALIV-SDB.ST', 'HEXA-B.ST', 'SKF-B.ST'],
            "G4": ['GETI-B.ST', 'HM-B.ST', 'EQNR.ST', 'EVO.ST', 'BOL.ST', 'ELUX-B.ST', 'EQT.ST', 'SINCH.ST']
        }

        all_targets = []

        for band in self.bands:
            logger.log(f"Bearbetar band {band}...")

            # Ladda bandfilen
            band_df = load_master(f"band_{band.lower()}_raw")

            # Filtrera PRICE på bandets tickers
            df_band = base[base["Ticker"].isin(bands[band])].copy()

            # Lägg till Band-kolumnen
            df = df_band.merge(band_df[["Date", "Ticker", "Band"]], on=["Date", "Ticker"], how="left")

            # rel_move + y
            df = self.compute_rel_move(df)
            df = self.ordinal_target(df)

            save_master(df, f"target_{band.lower()}")
            all_targets.append(df)

        # Skapa target_all
        target_all = pd.concat(all_targets).sort_values(["Ticker", "Date"])
        save_master(target_all, "target_all")

        logger.log("TargetBuilder klar.")




   
