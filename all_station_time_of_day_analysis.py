import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("combined_cleaned_near_accommodation.csv")

# -----------------------------
# Select years (combine 2022 and 2023)
# -----------------------------
years = [2022, 2023]
df = df[df["YEAR"].isin(years)]

# Convert TIME column to datetime
df["TIME"] = pd.to_datetime(df["TIME"])

# -----------------------------
# Fraction of bikes docked
# -----------------------------
df["frac_docked"] = df["AVAILABLE_BIKES"] / df["BIKE_STANDS"]

# -----------------------------
# Extract 30-minute time-of-day slot
# -----------------------------
df["time_of_day"] = df["TIME"].dt.strftime("%H:%M")

# -----------------------------
# Compute mean + variance across stations
# -----------------------------
# Group by time-of-day AND station
station_time = df.groupby(["time_of_day", "STATION ID"])["frac_docked"].mean()

# Now compute mean + variance across stations for each time slot
mean_frac = station_time.groupby("time_of_day").mean().sort_index()
variance_frac = station_time.groupby("time_of_day").var().sort_index()

# Reorder time-of-day from 05:00 -> 04:30 (next day)
times = pd.date_range(
    "2000-01-01 05:00",
    "2000-01-02 04:30",
    freq="30min"
)
desired_order = times.strftime("%H:%M")

mean_frac = mean_frac.reindex(desired_order)
variance_frac = variance_frac.reindex(desired_order)

# -----------------------------
# Plotting (Histogram with Variance Bars)
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    mean_frac.index,
    mean_frac.values,
    yerr=variance_frac.values,
    capsize=4,
    width=0.8,
    color="skyblue",
    edgecolor="black"
)

ax.set_xlabel("Time of Day (30-minute intervals)")
ax.set_ylabel("Mean Fraction of Bikes Docked")
ax.set_title(
    "Mean Fraction of Bikes Docked Across Stations Near Accommodation by Time of Day - 2022/2023 combined\n"
    "with variance between stations"
)

ax.set_ylim(0, 0.7)
plt.xticks(rotation=90)
plt.tight_layout()

# Save graph
os.makedirs("graphs", exist_ok=True)
path = "graphs/all_stations_mean_fraction_docked_near_accommodation_2022_2023_combined_adjusted.png"
fig.savefig(path)
plt.close(fig)

print(f"Saved system-wide histogram with variance bars: {path}")
