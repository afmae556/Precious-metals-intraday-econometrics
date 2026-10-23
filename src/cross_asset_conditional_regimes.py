import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
XAU_PATH = BASE_DIR / "data" / "XAUUSD_H1.csv"
XAG_PATH = BASE_DIR / "data" / "XAGUSD_H1.csv"

XAU_NAME = "XAUUSD"
XAG_NAME = "XAGUSD"

PLOTS_DIR = BASE_DIR / "plots"
TABLES_DIR = BASE_DIR / "tables"

NEUTRAL_THR = 0.05
LAGS = list(range(-6, 7))
WEEKDAY_FILTER = [0, 1, 2, 3, 4]


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_mt_like(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.replace("<", "").replace(">", "").strip().upper() for c in df.columns]
    req = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}.")
    dt = df["DATE"].astype(str).str.strip() + " " + df["TIME"].astype(str).str.strip()
    df["DATETIME"] = pd.to_datetime(dt, format="%Y.%m.%d %H:%M:%S", errors="raise")
    df = df.sort_values("DATETIME").reset_index(drop=True)
    df["RETURN_%"] = df["CLOSE"].pct_change() * 100.0
    df["HOUR"] = df["DATETIME"].dt.hour
    df["WEEKDAY"] = df["DATETIME"].dt.dayofweek
    return df.dropna(subset=["RETURN_%"])


def align_returns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    a = df_a[["DATETIME", "RETURN_%", "HOUR", "WEEKDAY"]].rename(columns={"RETURN_%": "RET_A"})
    b = df_b[["DATETIME", "RETURN_%"]].rename(columns={"RETURN_%": "RET_B"})
    merged = pd.merge(a, b, on="DATETIME", how="inner")
    if WEEKDAY_FILTER:
        merged = merged[merged["WEEKDAY"].isin(WEEKDAY_FILTER)]
    return merged.sort_values("DATETIME").reset_index(drop=True)


def sign_from_return(s: pd.Series, thr: float) -> pd.Series:
    return pd.Series(
        np.where(s > thr, "up", np.where(s < -thr, "down", "flat")),
        index=s.index,
    )


def lagged_xcorr(merged: pd.DataFrame, lags: list) -> pd.DataFrame:
    rows = []
    ret_a = merged["RET_A"].to_numpy()
    ret_b = merged["RET_B"].to_numpy()
    for lag in lags:
        if lag > 0:
            a, b = ret_a[lag:], ret_b[:-lag]
        elif lag < 0:
            L = -lag
            a, b = ret_a[:-L], ret_b[L:]
        else:
            a, b = ret_a, ret_b
        if len(a) < 30:
            corr = np.nan
        else:
            a_c, b_c = a - a.mean(), b - b.mean()
            corr = np.corrcoef(a_c, b_c)[0, 1]
        rows.append({"lag_h": lag, "corr": corr})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / f"xcorr_{XAU_NAME}_vs_{XAG_NAME}.csv", index=False)
    return out


def plot_xcorr(df_corr: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.stem(df_corr["lag_h"], df_corr["corr"])
    plt.axhline(0, ls="--", lw=1, color="gray")
    plt.title(f"Cross-Correlation Function (Lags) - {XAU_NAME} vs {XAG_NAME}")
    plt.xlabel("Lag (Hours, positive: XAG leads XAU)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"xcorr_{XAU_NAME}_vs_{XAG_NAME}.png", dpi=150)
    plt.close()


def conditional_hour_matrix(merged: pd.DataFrame, cond_lag: int = 0, thr: float = NEUTRAL_THR) -> pd.DataFrame:
    merged = merged.copy()
    merged["SIGN_A"] = sign_from_return(merged["RET_A"], thr)
    merged["SIGN_B"] = sign_from_return(merged["RET_B"], thr)
    merged["SIGN_B_LAG"] = merged["SIGN_B"].shift(cond_lag)
    df = merged.dropna(subset=["SIGN_A", "SIGN_B_LAG"])

    rows = []
    for h in range(24):
        sub = df[df["HOUR"] == h]
        if len(sub) == 0:
            continue
        uncond = sub["SIGN_A"].value_counts(normalize=True).reindex(["up", "down", "flat"]).fillna(0.0)
        for given in ["up", "down", "flat"]:
            s = sub[sub["SIGN_B_LAG"] == given]
            if len(s) == 0:
                continue
            cond = s["SIGN_A"].value_counts(normalize=True).reindex(["up", "down", "flat"]).fillna(0.0)
            rows.append({
                "hour": h,
                "given": given,
                "n": len(s),
                "P_up": cond["up"],
                "P_down": cond["down"],
                "P_flat": cond["flat"],
                "lift_up": cond["up"] / uncond["up"] if uncond["up"] > 0 else np.nan,
                "lift_down": cond["down"] / uncond["down"] if uncond["down"] > 0 else np.nan,
                "lift_flat": cond["flat"] / uncond["flat"] if uncond["flat"] > 0 else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / f"cond_hour_{XAU_NAME}_given_{XAG_NAME}.csv", index=False)
    return out


def plot_conditional(df_cond: pd.DataFrame):
    for given in ["up", "down", "flat"]:
        mat = df_cond[df_cond["given"] == given].set_index("hour")
        plt.figure(figsize=(10, 4))
        plt.plot(mat.index, mat["P_up"] * 100, marker="o", label="P(XAU = up)")
        plt.plot(mat.index, mat["P_down"] * 100, marker="o", label="P(XAU = down)")
        plt.plot(mat.index, mat["P_flat"] * 100, marker="o", label="P(XAU = flat)")
        plt.title(f"{XAU_NAME} Conditional Probability | {XAG_NAME} = {given}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Probability (%)")
        plt.xticks(range(0, 24))
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"cond_{XAU_NAME}_given_{XAG_NAME}_{given}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD vs XAGUSD Cross-Asset Conditional Econometrics")
    parser.add_argument("--cond-lag", type=int, default=0, help="Lag in hours for conditional regime transition (0=contemporaneous, 1=lead-lag)")
    args = parser.parse_args()

    ensure_dirs()
    df_xau = load_mt_like(XAU_PATH)
    df_xag = load_mt_like(XAG_PATH)

    merged = align_returns(df_xau, df_xag)
    print(f"Aligned dataset from {merged['DATETIME'].min()} to {merged['DATETIME'].max()} (n = {len(merged)} bars)")

    df_corr = lagged_xcorr(merged, LAGS)
    print("\n=== LAGGED CROSS-CORRELATION (XAUUSD vs XAGUSD) ===")
    print(df_corr.round(4).to_string(index=False))
    plot_xcorr(df_corr)

    df_cond = conditional_hour_matrix(merged, cond_lag=args.cond_lag)
    print(f"\n=== CONDITIONAL PROBABILITY MATRIX (LAG = {args.cond_lag}h, HEAD) ===")
    print(df_cond.head(9).round(4).to_string(index=False))
    plot_conditional(df_cond)

