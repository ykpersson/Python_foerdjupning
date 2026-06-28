import pandas as pd
import itertools
from support_functions import load_master, save_master, get_date_range


class PriceCombinatoricsBuilder:

    def __init__(self):
        # Synkad datumlogik
        self.tech_start, self.start, self.end = get_date_range()
        

        # Ladda master
        self.price = load_master("price_ohlc_raw")
        if self.price is None:
            raise FileNotFoundError("price_ohlc_raw.pkl saknas!")

        # Klipp bort pre-window
        self.price = self.price[self.price["Date"] >= self.tech_start].copy()

    def build(self):
        df = self.price.sort_values(["Ticker", "Date"]).copy()

        base_cols = ["Open", "High", "Low", "Close", "Volume"]

        out = df[["Date", "Ticker"]].copy()

        # 1. Ensamma variabler
        for col in base_cols:
            out[f"p_{col.lower()}"] = df[col]

        # Hjälpfunktioner
        def safe_div(a, b):
            return a / b.replace(0, pd.NA)

        def add(name, series):
            out[name] = series.replace([float("inf"), float("-inf")], pd.NA)

        # 2. 2-kombinationer
        for a, b in itertools.combinations(base_cols, 2):
            add(f"{a}_minus_{b}", df[a] - df[b])
            add(f"{a}_div_{b}", safe_div(df[a], df[b]))
            add(f"{a}_times_{b}", df[a] * df[b])

        # 3. 3-kombinationer
        for a, b, c in itertools.combinations(base_cols, 3):
            add(f"{a}_{b}_{c}_ratio", safe_div(df[a] - df[b], df[c]))
            add(f"{a}_{b}_{c}_prod", df[a] * df[b] * df[c])

        # 4. 4-kombinationer
        for a, b, c, d in itertools.combinations(base_cols, 4):
            add(f"{a}_{b}_{c}_{d}_ratio", safe_div(df[a] - df[b], df[c] * df[d]))
            add(f"{a}_{b}_{c}_{d}_prod", df[a] * df[b] * df[c] * df[d])

        # 5. 5-kombination
        a, b, c, d, e = base_cols
        add("all5_prod", df[a] * df[b] * df[c] * df[d] * df[e])
        add("all5_ratio", safe_div(df[a] - df[b], df[c] * df[d] * df[e]))

        # Rensa
        out = out.dropna(how="all", axis=1)
        out = out.dropna()

        save_master(out, "price_combinatorics_raw")
        print(f"Klar! Sparad som price_combinatorics_raw ({out.shape})")

        return out
