import pandas as pd
import logger
import setting
from support_functions import load_master, save_master, get_date_range, normalize_ticker


class BandBuilder:

    def __init__(self):
        self.tech_start, self.start, self.end = get_date_range()

        self.price = load_master("price_ohlc_raw")
        if self.price is None:
            raise FileNotFoundError("price_ohlc_raw.pkl saknas!")

        self.price["Ticker"] = self.price["Ticker"].apply(normalize_ticker)
        self.price = self.price[self.price["Ticker"].isin(setting.tickers)].copy()

        self.price["Date"] = pd.to_datetime(self.price["Date"])
        self.price = self.price[self.price["Date"] >= self.tech_start].copy()

    def build_all(self):
        logger.log("Startar BandBuilder..")

        df = self.price.sort_values(["Ticker", "Date"]).copy()

        df["Return"] = df.groupby("Ticker")["Close"].pct_change()
        df["Vol20"] = df.groupby("Ticker")["Return"].rolling(20).std().reset_index(0, drop=True)

        vol = df.groupby("Ticker")["Vol20"].median()
        vol = vol.fillna(vol.median())

        tickers_sorted = vol.sort_values().index.tolist()
        n = len(tickers_sorted)

        bands = {
            "G1": tickers_sorted[: n//4],
            "G2": tickers_sorted[n//4 : n//2],
            "G3": tickers_sorted[n//2 : 3*n//4],
            "G4": tickers_sorted[3*n//4 :]
        }

        # Skapa band-dataframes
        band_g1 = df[df["Ticker"].isin(bands["G1"])].copy()
        band_g2 = df[df["Ticker"].isin(bands["G2"])].copy()
        band_g3 = df[df["Ticker"].isin(bands["G3"])].copy()
        band_g4 = df[df["Ticker"].isin(bands["G4"])].copy()

        # Lägg till bandnamn
        band_g1["Band"] = "G1"
        band_g2["Band"] = "G2"
        band_g3["Band"] = "G3"
        band_g4["Band"] = "G4"

        # Spara
        save_master(band_g1, "band_g1_raw")
        save_master(band_g2, "band_g2_raw")
        save_master(band_g3, "band_g3_raw")
        save_master(band_g4, "band_g4_raw")

        logger.log("BandBuilder klar.")
        return bands
