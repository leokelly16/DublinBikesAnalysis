import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

#load data
df = pd.read_csv("combined_cleaned.csv")
df["TIME"] = pd.to_datetime(df["TIME"])
df = df[(df["TIME"] >= "2022-01-01") & (df["TIME"] <= "2023-12-31")]
df["date"] = df["TIME"].dt.date
df["weekday"] = df["TIME"].dt.weekday  # 0=Mon, 6=Sun

df["frac_not_docked"] = (df["BIKE_STANDS"] - df["AVAILABLE_BIKES"]) / df["BIKE_STANDS"]

# date range func
def d_range(s, e):
    return pd.date_range(start=s, end=e, freq="D").date

# date ranges according to the academic calendar
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

# catagorise data in to time periods
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

# catagories in rows
def assign_category(row):
    d = row["date"]
    wd = row["weekday"]
    
    if d in teaching_all:
        return "term_weekday" if wd < 5 else "term_weekend"
    if d in reading_all:
        return "reading_week"
    if d in scholarship_all:
        return "scholarship_exam"
    if d in christmas_exam_all:
        return "christmas_exam"
    if d in summer_exam_all:
        return "summer_exam"
    if d in christmas_closure_all:
        return "christmas_closure"
    if d in summer_all:
        return "summer"
    return "other_out_of_term"

df["category"] = df.apply(assign_category, axis=1)

os.makedirs("graphs", exist_ok=True)

# plotting func
def plot_availability(title, categories, filename):
    subset = df[df["category"].isin(categories)]
    grouped = subset.groupby(["STATION ID", "category"])["frac_not_docked"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Fraction of Bikes Not Docked (in use)")
    fig.tight_layout()
    fig.savefig(f"graphs/{filename}.png")
    plt.close(fig)

# make graphs

# 1. Term weekdays vs Summer
plot_availability(
    "Term Weekdays vs Summer (Bike Usage)",
    ["term_weekday", "summer"],
    "term_weekdays_vs_summer"
)

# 2. Term weekends vs Summer
plot_availability(
    "Term Weekends vs Summer (Bike Usage)",
    ["term_weekend", "summer"],
    "term_weekends_vs_summer"
)

# 3. Reading Week vs Term Weekdays
plot_availability(
    "Reading Week vs Term Weekdays",
    ["reading_week", "term_weekday"],
    "reading_week_vs_term"
)

# 4. Exam periods vs Term Weekdays
plot_availability(
    "Exam Periods vs Term Weekdays",
    ["scholarship_exam", "christmas_exam", "summer_exam", "term_weekday"],
    "exams_vs_term"
)

# 5. Christmas Closure vs Term Weekdays
plot_availability(
    "Christmas Closure vs Term Weekdays",
    ["christmas_closure", "term_weekday"],
    "christmas_closure_vs_term"
)

print("All graphs generated in ./graphs/")
