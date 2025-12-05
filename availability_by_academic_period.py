# Academic period availability analysis for Trinity-area stations (2022-2023)
# Focus: how academic calendar phases affect P(near-empty) and P(near-full) events.
# Proposed question: How do commute periods vs off‑peak times and bank holidays change the probability that a Trinity‑area station is “near empty” (≤2 bikes) or “near full” (≤2 free stands)? This targets an applied‑probability event (low/high availability) and compares conditional probabilities across time/day types.

# Approach:

# Label each timestamp with hour bucket (morning peak 7–10, midday 11–15, evening peak 16–19, night 20–6), day type (weekday/weekend), and Irish bank holidays (2022–2023 list embedded in code; no external download).
# Compute event probabilities and 95% CIs; do two‑proportion z‑tests for peaks vs off‑peak and holidays vs regular weekdays.
# Plot bars for probability of near‑empty by hour (weekday vs weekend vs holiday) and by station for peak vs off‑peak; repeat for near‑full.

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# Event thresholds
NEAR_EMPTY_THRESHOLD = 2      # bikes remaining
NEAR_FULL_THRESHOLD = 2       # free stands remaining


# Helper to build date ranges
def d_range(start, end):
    return pd.date_range(start=start, end=end, freq="D").date


# Academic calendar date ranges (copied from docked_analysis_according to_academic_calendar.py)
scholarship_2022 = d_range("2022-01-10", "2022-01-17")
teaching_2022_block1 = d_range("2022-01-24", "2022-03-06")
reading_week_2022 = d_range("2022-03-07", "2022-03-14")
teaching_2022_block2 = d_range("2022-03-15", "2022-05-01")
summer_exams_2022 = d_range("2022-05-02", "2022-05-09")
term_end_2022 = d_range("2022-05-30", "2022-06-05")
summer_2022 = d_range("2022-06-06", "2022-09-11")
teaching_2022_2023_block1 = d_range("2022-09-12", "2022-10-23")
reading_week_2022_2023 = d_range("2022-10-24", "2022-11-01")
teaching_2022_2023_block2 = d_range("2022-11-02", "2022-12-11")
christmas_exams_2022 = d_range("2022-12-12", "2022-12-18")
christmas_closure_2022 = d_range("2022-12-23", "2023-01-02")
preterm_study_2023 = d_range("2023-01-03", "2023-01-08")
scholarship_2023 = d_range("2023-01-09", "2023-01-16")
teaching_2023_block1 = d_range("2023-01-23", "2023-03-05")
reading_week_2023 = d_range("2023-03-06", "2023-03-13")
teaching_2023_block2 = d_range("2023-03-14", "2023-04-30")
summer_exams_2023 = d_range("2023-05-01", "2023-05-08")
term_end_2023 = d_range("2023-05-29", "2023-06-04")
summer_2023 = d_range("2023-06-05", "2023-09-10")
teaching_2023_2024_block1 = d_range("2023-09-11", "2023-10-22")
reading_week_2023_2024 = d_range("2023-10-23", "2023-11-01")
teaching_2023_2024_block2 = d_range("2023-11-02", "2023-12-03")
revision_2023 = d_range("2023-12-04", "2023-12-10")
christmas_exams_2023 = d_range("2023-12-11", "2023-12-17")
christmas_closure_2023 = d_range("2023-12-22", "2023-12-31")

# Collections
teaching_all = (
    set(teaching_2022_block1)
    | set(teaching_2022_block2)
    | set(teaching_2022_2023_block1)
    | set(teaching_2022_2023_block2)
    | set(teaching_2023_block1)
    | set(teaching_2023_block2)
    | set(teaching_2023_2024_block1)
    | set(teaching_2023_2024_block2)
)
reading_all = set(reading_week_2022) | set(reading_week_2022_2023) | set(reading_week_2023) | set(reading_week_2023_2024)
scholarship_all = set(scholarship_2022) | set(scholarship_2023)
summer_exam_all = set(summer_exams_2022) | set(summer_exams_2023)
christmas_exam_all = set(christmas_exams_2022) | set(christmas_exams_2023)
christmas_closure_all = set(christmas_closure_2022) | set(christmas_closure_2023)
summer_all = set(summer_2022) | set(summer_2023)


def assign_period(row_date, weekday):
    """Assign a broad academic period; suffix weekday/weekend for main buckets."""
    if row_date in teaching_all:
        return "teaching_weekday" if weekday < 5 else "teaching_weekend"
    if row_date in reading_all:
        return "reading_week"
    if row_date in scholarship_all:
        return "scholarship_exam"
    if row_date in christmas_exam_all:
        return "christmas_exam"
    if row_date in summer_exam_all:
        return "summer_exam"
    if row_date in christmas_closure_all:
        return "christmas_closure"
    if row_date in summer_all:
        return "summer_weekday" if weekday < 5 else "summer_weekend"
    return "other_out_of_term"


def proportion_ci(count: int, n: int):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = count / n
    z = 1.96
    se = math.sqrt(p * (1 - p) / n)
    return p, max(0, p - z * se), min(1, p + z * se)


def compute_probabilities(df: pd.DataFrame, group_col: str, event_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col)[event_col].agg(["sum", "count"]).reset_index()
    grouped[["prob", "ci_low", "ci_high"]] = grouped.apply(
        lambda r: proportion_ci(r["sum"], r["count"]), axis=1, result_type="expand"
    )
    grouped.rename(columns={"sum": "event_count", "count": "n"}, inplace=True)
    return grouped


def plot_probabilities(df: pd.DataFrame, title: str, ylabel: str, filename: str):
    ordered = [
        "teaching_weekday",
        "teaching_weekend",
        "reading_week",
        "scholarship_exam",
        "summer_exam",
        "christmas_exam",
        "christmas_closure",
        "summer_weekday",
        "summer_weekend",
        "other_out_of_term",
    ]
    df = df.set_index("period").reindex(ordered).dropna(subset=["prob"])
    ax = df["prob"].plot(kind="bar", figsize=(10, 6), color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0, min(1, df["prob"].max() * 1.15 + 0.02))
    # optional error bars
    ax.errorbar(range(len(df)), df["prob"], yerr=[df["prob"] - df["ci_low"], df["ci_high"] - df["prob"]], fmt="none", ecolor="black", capsize=4)
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(os.path.join("graphs", filename))
    plt.close()


def main():
    df = pd.read_csv("combined_cleaned_near_accommodation.csv")
    df["TIME"] = pd.to_datetime(df["TIME"])
    df["date"] = df["TIME"].dt.date
    df["weekday_num"] = df["TIME"].dt.weekday  # 0=Mon

    df["period"] = df.apply(lambda r: assign_period(r["date"], r["weekday_num"]), axis=1)
    df["near_empty"] = df["AVAILABLE_BIKES"] <= NEAR_EMPTY_THRESHOLD
    df["near_full"] = df["AVAILABLE_BIKE_STANDS"] <= NEAR_FULL_THRESHOLD

    near_empty_prob = compute_probabilities(df, "period", "near_empty")
    near_full_prob = compute_probabilities(df, "period", "near_full")

    plot_probabilities(
        near_empty_prob,
        title="P(near-empty) by Academic Period near Accommodation",
        ylabel=f"P(available bikes <= {NEAR_EMPTY_THRESHOLD})",
        filename="near_empty_by_academic_period_near_accommodation.png",
    )
    plot_probabilities(
        near_full_prob,
        title="P(near-full) by Academic Period near Accommodation",
        ylabel=f"P(free stands <= {NEAR_FULL_THRESHOLD})",
        filename="near_full_by_academic_period_near_accommodation.png",
    )

    print("Saved academic-period availability graphs to ./graphs/")


if __name__ == "__main__":
    main()
