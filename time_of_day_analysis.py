import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv('combined_cleaned.csv')

# -----------------------------
# Select station + year
# -----------------------------
station_id = 21   # change as needed
year = 2023

station_data = df[(df['STATION ID'] == station_id) & (df['YEAR'] == year)]

# Convert TIME column to datetime
station_data['TIME'] = pd.to_datetime(station_data['TIME'])

# -----------------------------
# Fraction of bikes docked
# -----------------------------
station_data["frac_docked"] = (
    station_data["AVAILABLE_BIKES"] / station_data["BIKE_STANDS"]
)

# -----------------------------
# Extract 30‑minute time-of-day slot
# -----------------------------
station_data["time_of_day"] = station_data["TIME"].dt.strftime("%H:%M")

# -----------------------------
# Compute mean fraction docked for each 30‑minute slot
# -----------------------------
mean_frac = station_data.groupby("time_of_day")["frac_docked"].mean()

# Sort by actual time order
mean_frac = mean_frac.sort_index()

# -----------------------------
# Plotting (Histogram / Bar Chart)
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(mean_frac.index, mean_frac.values, width=0.8)

ax.set_xlabel("Time of Day (30‑minute intervals)")
ax.set_ylabel("Mean Fraction of Bikes Docked")
ax.set_title(f"Mean Fraction of Bikes Docked by 30‑Minute Interval\nStation {station_id} — Year {year}")

plt.xticks(rotation=90)
plt.tight_layout()

# Save graph
os.makedirs("graphs", exist_ok=True)
path = f"graphs/station_{station_id}_mean_fraction_docked_{year}.png"
fig.savefig(path)
plt.close(fig)

print(f"Saved yearly histogram: {path}")