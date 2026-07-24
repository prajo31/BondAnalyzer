"""Pull the U.S. Treasury par-yield curve from FRED."""
import numpy as np
import pandas as pd
from fredapi import Fred

# label -> (FRED series id, maturity in years)
TENORS = {
    "1M": ("DGS1MO", 1 / 12), "3M": ("DGS3MO", 0.25),
    "6M": ("DGS6MO", 0.5), "1Y": ("DGS1", 1),
    "2Y": ("DGS2", 2), "3Y": ("DGS3", 3),
    "5Y": ("DGS5", 5), "7Y": ("DGS7", 7),
    "10Y": ("DGS10", 10), "20Y": ("DGS20", 20),
    "30Y": ("DGS30", 30),
}


def get_yield_curve(api_key):
    """Latest Treasury yields by tenor (percent) as a DataFrame."""
    fred = Fred(api_key=api_key)
    rows = []
    for label, (series, yrs) in TENORS.items():
        s = fred.get_series(series).dropna()
        if not s.empty:
            rows.append({"tenor": label, "years": yrs,
                         "yield": float(s.iloc[-1])})
    return pd.DataFrame(rows)


def matched_treasury_yield(curve, years):
    """Interpolate the Treasury yield (percent) for a maturity."""
    c = curve.sort_values("years")
    return float(np.interp(years, c["years"], c["yield"]))
