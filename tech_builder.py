import pandas as pd
import logger
import setting
from support_functions import save_master, load_master, get_date_range, normalize_ticker

def compute_technical_features(df):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # === 1. Daily Return ===
    df["ret"] = df["Close"].pct_change()

    # === 2. Volatility (20-day std) ===
    df["vol_20"] = df["ret"].rolling(20).std()

    # === 3. Momentum (10-day return) ===
    df["momentum_10"] = df["Close"].pct_change(10)

    # === 4. RSI (14) ===
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # === 5. MACD (12,26) ===
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # === 6. MA10 ===
    df["ma10"] = df["Close"].rolling(10).mean()

    # === 7. MA20 + MA-cross ===
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma_cross"] = (df["ma10"] > df["ma20"]).astype(int)

    # === 8. Bollinger Band Width (20) ===
    mean20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = mean20 + 2 * std20
    lower = mean20 - 2 * std20
    df["bb_width"] = (upper - lower) / mean20

    # === 9. ATR (14) ===
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # === 10. Stochastic %K (14) ===
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (df["Close"] - low14) / (high14 - low14)

    return df

class TechBuilder:
    """Modern teknisk feature-builder för ALLA tickers."""

    def __init__(self):
        self.tickers = setting.tickers

    def build_all(self):
        logger.log("Startar TechBuilderV4...")

        # 1. Läs rätt fil
        price = load_master("price_ohlc_raw")

        # 2. Normalisera ticker direkt
        price["Ticker"] = price["Ticker"].apply(normalize_ticker)

        all_frames = []

        for ticker in self.tickers:
            # 3. Filtrera på rätt kolumnnamn
            df = price[price["Ticker"] == ticker].copy()

            if df.empty:
                logger.log(f"TechBuilder: Ingen data för {ticker}")
                continue

            # 4. Beräkna tekniska features
            tech = compute_technical_features(df)

            # 5. Sätt korrekt tickerkolumn
            tech["Ticker"] = ticker

            all_frames.append(tech)

        tech_all = pd.concat(all_frames, ignore_index=True)

        # 6. Rensa NaN / Inf
        tech_all = tech_all.replace([float("inf"), float("-inf")], pd.NA)
        tech_all = tech_all.dropna()

        # 7. Spara med rätt namn
        save_master(tech_all, "tech_indicators_raw")

        logger.log(f"TechBuilder klar ({tech_all.shape})")

        return tech_all

