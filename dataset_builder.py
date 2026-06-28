import pandas as pd
import logger
from support_functions import load_master, save_master, get_date_range


class DatasetBuilder:

    def __init__(self):
        self.tech_start, self.start, self.end = get_date_range()

        # === Fyra masterfiler ===
        self.price_raw_path = "price_ohlc_raw"
        self.price_feat_path = "price_combinatorics_raw"
        self.tech_path = "tech_indicators_raw"
        self.target_path = "target_all"

    def build_all(self):

        logger.log("Startar DatasetBuilder...")

        # === PRICE RAW ===
        price_raw = load_master(self.price_raw_path)
        price_raw = price_raw[price_raw["Date"] >= self.tech_start].copy()
        price_raw = price_raw.sort_values(["Ticker", "Date"])

        # === PRICE FEATURES (Return_h etc.) ===
        price_feat = load_master(self.price_feat_path)
        price_feat = price_feat[price_feat["Date"] >= self.tech_start].copy()
        price_feat = price_feat.sort_values(["Ticker", "Date"])

        # === TECH INDICATORS ===
        tech = load_master(self.tech_path)
        tech = tech[tech["Date"] >= self.tech_start].copy()
        tech = tech.sort_values(["Ticker", "Date"])

        # === TARGETS ===
        targets = load_master(self.target_path)
        targets = targets[targets["Date"] >= self.tech_start].copy()
        targets = targets.sort_values(["Ticker", "Date"])

        # === MERGE PRICE RAW + PRICE FEATURES ===
        df = price_raw.merge(
            price_feat,
            on=["Date", "Ticker"],
            how="left"
        )

        # === MERGE TECH ===
        df = df.merge(
            tech,
            on=["Date", "Ticker"],
            how="left"
        )

        # === MERGE TARGET ===
        df = df.merge(
            targets,
            on=["Date", "Ticker"],
            how="left"
        )

        # === CLEAN ===
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.dropna()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # === SAVE ===
        save_master(df, "dataset_all")

        logger.log(f"DatasetBuilder klar ({df.shape})")

        return df
