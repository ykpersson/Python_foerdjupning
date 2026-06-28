import yfinance as yf
import pandas as pd
import logger
from support_functions import save_master, get_date_range, normalize_ticker, load_master


class IndexBuilder:

    def __init__(self):
        self.ticker = "^OMX"
        self.tech_start, self.start, self.end = get_date_range()

    def fetch(self):
        """Hämtar rå indexdata från Yahoo Finance."""
        logger.log(f"Hämtar indexdata för OMXS30 från {self.start.date()} till {self.end.date()}...")

        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval="1d",
            auto_adjust=False
        ).reset_index()

        # Platta MultiIndex-kolumner från yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = normalize_ticker(self.ticker)

        return df

    def align_to_price_dates(self, index_df):
        """Alignar index mot datum i price_ohlc_raw."""
        price = load_master("price_ohlc_raw")
        all_dates = pd.DataFrame({"Date": pd.to_datetime(price["Date"].unique())})

        # Merge: alla datum från price, index där det finns
        df = all_dates.merge(index_df, on="Date", how="left")

        # Ta bort datum där index saknas (trasiga rader)
        df = df[df["Close"].notna()].copy()

        return df

    def build_one(self, df):
        """Beräknar indexets dagliga rörelse."""
        df = df.reset_index(drop=True)
        df["Index_h"] = df["Close"].pct_change()
        return df[["Date", "Index_h"]]

    def build_all(self):
        logger.log("Startar IndexBuilder...")

        raw_index = self.fetch()
        aligned_index = self.align_to_price_dates(raw_index)
        final_index = self.build_one(aligned_index)

        save_master(final_index, "index_all")
        logger.log(f"IndexBuilder klar ({final_index.shape})")

        return final_index

