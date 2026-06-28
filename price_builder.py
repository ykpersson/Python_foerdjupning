import pandas as pd
import yfinance as yf
import setting
from support_functions import save_master, get_date_range,normalize_ticker
import logger


class PriceBuilderV3:

    def __init__(self):
        self.tickers = setting.tickers
        self.tech_start, self.start, self.end = get_date_range()


    def fetch_price(self, ticker):
        logger.log(f"Hämtar {ticker} från {self.start.date()} till {self.end.date()}")

        df = yf.download(
            ticker,
            start=self.start,
            end=self.end,
            interval="1d",
            auto_adjust=False
        )

        if df.empty:
            return pd.DataFrame()

        # --- FIX 1: Flatten MultiIndex från yfinance ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()

        # --- FIX 2: Säkerställ rätt kolumnnamn ---
        rename_map = {
            "Adj Close": "Adj Close",
            "Close": "Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Volume": "Volume"
        }
        df = df.rename(columns=rename_map)

        # --- FIX 3: Return_h ---
        df["Ticker"] = ticker
        df["Return_h"] = df["Close"].pct_change()
        df["Ticker"] = df["Ticker"].apply(normalize_ticker)

        return df

    def build_all(self):
        logger.log("Startar PriceBuilderV3...")

        frames = []
        for ticker in self.tickers:
            df = self.fetch_price(ticker)
            if df.empty:
                logger.log(f"Ingen data för {ticker}")
                continue
            frames.append(df)

        if not frames:
            logger.log("PriceBuilderV3: Ingen data att spara.")
            return pd.DataFrame()

        price_all = pd.concat(frames, ignore_index=True)
        price_all = price_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # FIX: flatten kolumn-MultiIndex och sen spara
        if isinstance(price_all.columns, pd.MultiIndex):
            price_all.columns = [
            "_".join([str(c) for c in col]).strip("_")
            for col in price_all.columns
    ]

        save_master(price_all, "price_ohlc_raw")
        logger.log(f"PriceBuilderV3 klar ({len(price_all)} rader).")

        return price_all
