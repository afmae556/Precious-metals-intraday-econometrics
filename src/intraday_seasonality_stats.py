import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "XAUUSD_H1.csv"
PLOTS_DIR = BASE_DIR / "plots"
TABLES_DIR = BASE_DIR / "tables"
NEUTRAL_THR = 0.05


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def rate(cond):
    return lambda x: 100.0 * np.mean(cond(x))


def load_xauusd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.replace("<", "").replace(">", "").strip().upper() for c in df.columns]

    required = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}.")

    dt = df["DATE"].astype(str).str.strip() + " " + df["TIME"].astype(str).str.strip()
    df["DATETIME"] = pd.to_datetime(dt, format="%Y.%m.%d %H:%M:%S", errors="raise")
    df = df.sort_values("DATETIME").reset_index(drop=True)

    df["RETURN_%"] = df["CLOSE"].pct_change() * 100.0
    df["HOUR"] = df["DATETIME"].dt.hour
    df["WEEKDAY"] = df["DATETIME"].dt.dayofweek
    return df.dropna(subset=["RETURN_%", "HOUR"])


def hourly_stats(df: pd.DataFrame, neutral_thr: float = 0.05) -> pd.DataFrame:
    stats = (
        df.groupby("HOUR")["RETURN_%"]
        .agg(
            mean_return="mean",
            median_return="median",
            std_return="std",
            positive_rate=rate(lambda x: x > 0),
            negative_rate=rate(lambda x: x < 0),
            neutral_rate=rate(lambda x: x.abs() < neutral_thr),
            n_obs="count",
        )
        .reset_index()
        .sort_values("HOUR")
    )
    return stats


def plot_hourly_returns(stats: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(stats["HOUR"], stats["mean_return"], marker="o", label="Mean Return (%)")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(24))
    plt.title("XAUUSD (1H) - Mean Return by Hour of Day")
    plt.xlabel("Hour of Day (Dataset Timezone)")
    plt.ylabel("Mean Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "hourly_mean_return.png", dpi=150)
    plt.close()


def plot_hourly_probs(stats: pd.DataFrame, neutral_thr: float, outdir: Path):
    plt.figure(figsize=(12, 6))
    width = 0.35
    plt.bar(stats["HOUR"] - width / 2, stats["positive_rate"], width=width, label="Positive Return Rate (%)")
    plt.bar(stats["HOUR"] + width / 2, stats["negative_rate"], width=width, label="Negative Return Rate (%)")
    plt.plot(
        stats["HOUR"],
        stats["neutral_rate"],
        marker="s",
        label=f"Neutral Regime Rate (|ret| < {neutral_thr}%)",
    )
    plt.xticks(range(24))
    plt.title("XAUUSD (1H) - Directional Regime Probabilities by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Probability (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "hourly_probabilities.png", dpi=150)
    plt.close()


def heatmaps_hour_x_weekday(df: pd.DataFrame, outdir: Path):
    dff = df[df["WEEKDAY"] <= 4].copy()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    pivot_mean = (
        dff.pivot_table(index="WEEKDAY", columns="HOUR", values="RETURN_%", aggfunc="mean")
        .reindex(index=[0, 1, 2, 3, 4])
    )
    plt.figure(figsize=(14, 5))
    im = plt.imshow(pivot_mean, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, label="Mean Return (%)")
    plt.yticks(range(5), days)
    plt.xticks(range(24))
    plt.title("XAUUSD (1H) - Mean Return Heatmap (Weekday x Hour)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_mean.png", dpi=150)
    plt.close()

    dff["IS_UP"] = (dff["RETURN_%"] > 0).astype(float) * 100.0
    pivot_up = (
        dff.pivot_table(index="WEEKDAY", columns="HOUR", values="IS_UP", aggfunc="mean")
        .reindex(index=[0, 1, 2, 3, 4])
    )
    plt.figure(figsize=(14, 5))
    im2 = plt.imshow(pivot_up, aspect="auto", cmap="Blues")
    plt.colorbar(im2, label="Positive Return Frequency (%)")
    plt.yticks(range(5), days)
    plt.xticks(range(24))
    plt.title("XAUUSD (1H) - Positive Return Probability Heatmap (Weekday x Hour)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_up.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ensure_dirs()
    df_xau = load_xauusd(DATA_PATH)
    stats_df = hourly_stats(df_xau, NEUTRAL_THR)
    print("=== XAUUSD HOURLY SEASONALITY SUMMARY ===")
    print(stats_df.round(4).to_string(index=False))
    plot_hourly_returns(stats_df, PLOTS_DIR)
    plot_hourly_probs(stats_df, NEUTRAL_THR, PLOTS_DIR)
    heatmaps_hour_x_weekday(df_xau, PLOTS_DIR)
    print(f"\nSaved intraday seasonality figures to: {PLOTS_DIR}")

