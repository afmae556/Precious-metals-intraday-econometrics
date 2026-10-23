import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "XAUUSD_H1.csv"
PLOTS_DIR = BASE_DIR / "plots"
TABLES_DIR = BASE_DIR / "tables"
PIP_SIZE = 0.01
TOP_K = 3


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_xauusd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.replace("<", "").replace(">", "").strip().upper() for c in df.columns]
    req = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns {miss}.")

    dt = df["DATE"].astype(str).str.strip() + " " + df["TIME"].astype(str).str.strip()
    df["DATETIME"] = pd.to_datetime(dt, format="%Y.%m.%d %H:%M:%S", errors="raise")
    df = df.sort_values("DATETIME").reset_index(drop=True)

    df["HOUR"] = df["DATETIME"].dt.hour
    df["WEEKDAY"] = df["DATETIME"].dt.dayofweek
    df["MONTH"] = df["DATETIME"].dt.month

    df["RETURN_%"] = df["CLOSE"].pct_change() * 100.0
    df["PIPS_MOVE"] = (df["CLOSE"] - df["CLOSE"].shift(1)) / PIP_SIZE
    df["PIPS_ABS"] = df["PIPS_MOVE"].abs()
    df["PIPS_RANGE"] = (df["HIGH"] - df["LOW"]) / PIP_SIZE

    return df.dropna(subset=["RETURN_%", "PIPS_MOVE", "PIPS_ABS", "PIPS_RANGE"])


def analyze_top_weekday_hour_pairs(df: pd.DataFrame, top_k: int = 3):
    dff = df[df["WEEKDAY"] <= 4].copy()
    grp = (
        dff.groupby(["WEEKDAY", "HOUR"])
        .agg(
            mean_return=("RETURN_%", "mean"),
            pips_abs_mean=("PIPS_ABS", "mean"),
            pips_range_mean=("PIPS_RANGE", "mean"),
            n_obs=("RETURN_%", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["n_obs"] >= 15]
    best_up = grp.sort_values("mean_return", ascending=False).head(top_k)
    best_down = grp.sort_values("mean_return", ascending=True).head(top_k)

    best_up[["WEEKDAY", "HOUR", "mean_return", "n_obs"]].to_csv(TABLES_DIR / "best_up_pairs.csv", index=False)
    best_down[["WEEKDAY", "HOUR", "mean_return", "n_obs"]].to_csv(TABLES_DIR / "best_down_pairs.csv", index=False)

    monthly = (
        dff.groupby("MONTH")
        .agg(
            pips_abs_mean=("PIPS_ABS", "mean"),
            pips_abs_median=("PIPS_ABS", "median"),
            pips_range_mean=("PIPS_RANGE", "mean"),
            pips_range_median=("PIPS_RANGE", "median"),
            n_obs=("PIPS_ABS", "count"),
        )
        .reset_index()
    )
    monthly.to_csv(TABLES_DIR / "monthly_pips_overall.csv", index=False)
    return best_up, best_down, monthly


if __name__ == "__main__":
    ensure_dirs()
    df = load_xauusd(DATA_PATH)
    up, down, monthly = analyze_top_weekday_hour_pairs(df, TOP_K)
    print("=== TOP BULLISH (WEEKDAY, HOUR) WINDOWS ===")
    print(up.round(4).to_string(index=False))
    print("\n=== TOP BEARISH (WEEKDAY, HOUR) WINDOWS ===")
    print(down.round(4).to_string(index=False))
    print("\n=== MONTHLY VOLATILITY IN PIPS ===")
    print(monthly.round(2).to_string(index=False))

