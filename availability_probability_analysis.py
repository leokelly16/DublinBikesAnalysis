# Proposed question: How do commute periods vs off‑peak times and bank holidays change the probability that a Trinity‑area station is “near empty” (≤2 bikes) or “near full” (≤2 free stands)? This targets an applied‑probability event (low/high availability) and compares conditional probabilities across time/day types.

# Approach:

# Label each timestamp with hour bucket (morning peak 7–10, midday 11–15, evening peak 16–19, night 20–6), day type (weekday/weekend), and Irish bank holidays (2022–2023 list embedded in code; no external download).
# Compute event probabilities and 95% CIs; do two‑proportion z‑tests for peaks vs off‑peak and holidays vs regular weekdays.
# Plot bars for probability of near‑empty by hour (weekday vs weekend vs holiday) and by station for peak vs off‑peak; repeat for near‑full.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, date

# Thresholds for events of interest
NEAR_EMPTY_THRESHOLD = 2  # bikes remaining
NEAR_FULL_THRESHOLD = 2   # free stands remaining

# Hour bins for temporal analysis
HOUR_BINS = [
    ("morning_peak", 7, 10),
    ("midday", 11, 15),
    ("evening_peak", 16, 19),
    ("night", 20, 23),
    ("overnight", 0, 6),
]

# Irish bank holidays for 2022-2023 (inclusive)
BANK_HOLIDAYS = {
    # 2022
    date(2022, 1, 3),  # New Year (observed)
    date(2022, 3, 17), # St Patrick's Day
    date(2022, 3, 18), # 2022 one-off bank holiday
    date(2022, 4, 18), # Easter Monday
    date(2022, 5, 2),  # May Day
    date(2022, 6, 6),  # June bank holiday
    date(2022, 8, 1),  # August bank holiday
    date(2022, 10, 31),# October bank holiday
    date(2022, 12, 26),# St Stephen's Day
    date(2022, 12, 27),# Christmas (observed)
    # 2023
    date(2023, 1, 2),
    date(2023, 3, 17),
    date(2023, 4, 10),
    date(2023, 5, 1),
    date(2023, 6, 5),
    date(2023, 8, 7),
    date(2023, 10, 30),
    date(2023, 12, 25),
    date(2023, 12, 26),
}


def assign_hour_bin(ts: pd.Timestamp) -> str:
    """Map an hour to a named bin."""
    hour = ts.hour
    for name, start, end in HOUR_BINS:
        if start <= end and start <= hour <= end:
            return name
        if start > end and (hour >= start or hour <= end):
            return name
    return "other"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TIME"] = pd.to_datetime(df["TIME"])
    df["date"] = df["TIME"].dt.date
    df["hour_bin"] = df["TIME"].apply(assign_hour_bin)
    df["day_category"] = np.where(df["TIME"].dt.weekday < 5, "weekday", "weekend")
    df.loc[df["date"].isin(BANK_HOLIDAYS), "day_category"] = "bank_holiday"
    df["peak_status"] = np.where(df["hour_bin"].isin(["morning_peak", "evening_peak"]), "peak", "off_peak")
    df["near_empty"] = df["AVAILABLE_BIKES"] <= NEAR_EMPTY_THRESHOLD
    df["near_full"] = df["AVAILABLE_BIKE_STANDS"] <= NEAR_FULL_THRESHOLD
    return df


def proportion_ci(count: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = count / n
    z = 1.96  # for ~95% CI
    se = np.sqrt(p * (1 - p) / n)
    return p, max(0, p - z * se), min(1, p + z * se)


def compute_probabilities(df: pd.DataFrame, group_cols, event_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[event_col].agg(["sum", "count"]).reset_index()
    grouped[["prob", "ci_low", "ci_high"]] = grouped.apply(
        lambda r: proportion_ci(r["sum"], r["count"]), axis=1, result_type="expand"
    )
    grouped.rename(columns={"sum": "event_count", "count": "n"}, inplace=True)
    return grouped


def two_proportion_ztest(count1: int, n1: int, count2: int, n2: int):
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = count1 / n1, count2 / n2
    pooled = (count1 + count2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    # two-sided p-value from normal approximation
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def plot_probability_bar(df: pd.DataFrame, x: str, hue: str, value: str, title: str, ylabel: str, filename: str):
    pivot = df.pivot(index=x, columns=hue, values=value).fillna(0)
    pivot = pivot.reindex(index=[b[0] for b in HOUR_BINS if b[0] in pivot.index], columns=sorted(pivot.columns))
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title=hue)
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", filename))
    plt.close()


def main():
    df = pd.read_csv("combined_cleaned_near_accommodation.csv")
    df = add_time_features(df)

    # Probabilities by hour bin and day type
    near_empty_hour = compute_probabilities(df, ["hour_bin", "day_category"], "near_empty")
    near_full_hour = compute_probabilities(df, ["hour_bin", "day_category"], "near_full")

    # Station-level peak vs off-peak probabilities
    near_empty_station_peak = compute_probabilities(df, ["STATION ID", "peak_status"], "near_empty")
    near_full_station_peak = compute_probabilities(df, ["STATION ID", "peak_status"], "near_full")

    # Aggregate peak vs off-peak (all stations combined)
    peak_vs_off = compute_probabilities(df, ["peak_status"], "near_empty")
    weekday_vs_holiday = compute_probabilities(df[df["day_category"].isin(["weekday", "bank_holiday"])], ["day_category"], "near_empty")

    # Hypothesis tests
    if len(peak_vs_off) == 2:
        c1, n1 = peak_vs_off.loc[peak_vs_off["peak_status"] == "peak", ["event_count", "n"]].values[0]
        c2, n2 = peak_vs_off.loc[peak_vs_off["peak_status"] == "off_peak", ["event_count", "n"]].values[0]
        z, p = two_proportion_ztest(c1, n1, c2, n2)
        print(f"Near-empty: peak vs off-peak z={z:.3f}, p={p:.4f} (counts {c1}/{n1} vs {c2}/{n2})")

    if len(weekday_vs_holiday) == 2:
        c1, n1 = weekday_vs_holiday.loc[weekday_vs_holiday["day_category"] == "weekday", ["event_count", "n"]].values[0]
        c2, n2 = weekday_vs_holiday.loc[weekday_vs_holiday["day_category"] == "bank_holiday", ["event_count", "n"]].values[0]
        z, p = two_proportion_ztest(c1, n1, c2, n2)
        print(f"Near-empty: weekday vs bank holiday z={z:.3f}, p={p:.4f} (counts {c1}/{n1} vs {c2}/{n2})")

    # Plots
    plot_probability_bar(
        near_empty_hour,
        x="hour_bin",
        hue="day_category",
        value="prob",
        title="Probability of Near-Empty Stations by Time of Day near Accommodation",
        ylabel="P(available bikes <= 2)",
        filename="near_empty_by_hour_daytype_near_accommodation.png",
    )

    plot_probability_bar(
        near_full_hour,
        x="hour_bin",
        hue="day_category",
        value="prob",
        title="Probability of Near-Full Stations by Time of Day near Accommodation",
        ylabel="P(free stands <= 2)",
        filename="near_full_by_hour_daytype_near_accommodation.png",
    )

    # Station-level peak vs off-peak bar charts (one for near empty, one for near full)
    for data, event_label, fname in [
        (near_empty_station_peak, "P(near empty)", "near_empty_peak_vs_off_by_station_near_accommodation.png"),
        (near_full_station_peak, "P(near full)", "near_full_peak_vs_off_by_station_near_accommodation.png"),
    ]:
        pivot = data.pivot(index="STATION ID", columns="peak_status", values="prob").fillna(0)
        pivot = pivot[sorted(pivot.columns)]
        ax = pivot.plot(kind="bar", figsize=(10, 6))
        ax.set_title(f"{event_label} by Station: Peak vs Off-Peak near Accommodation")
        ax.set_ylabel(event_label)
        ax.set_xlabel("Station ID")
        ax.legend(title="peak_status")
        plt.tight_layout()
        os.makedirs("graphs", exist_ok=True)
        plt.savefig(os.path.join("graphs", fname))
        plt.close()

    print("Graphs saved to ./graphs. Run this script inside your virtual environment to refresh outputs.")


if __name__ == "__main__":
    main()
