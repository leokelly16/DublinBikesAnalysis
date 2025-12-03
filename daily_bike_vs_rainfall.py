import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Bike usage data ---
df = pd.read_csv('combined_cleaned.csv')

station_id = 21   # replace with actual station ID
year = 2023       # match your dataset year
station_data = df[(df['STATION ID'] == station_id) & (df['YEAR'] == year)]
station_data['TIME'] = pd.to_datetime(station_data['TIME'])

# Filter only January
station_data = station_data[station_data['TIME'].dt.month == 1]

# Create a daily column
station_data['day'] = station_data['TIME'].dt.date

# Fraction of bikes not docked (in use)
station_data["frac_not_docked"] = (
    (station_data["BIKE_STANDS"] - station_data["AVAILABLE_BIKES"]) / station_data["BIKE_STANDS"]
)

# Daily summary for January
daily_summary = station_data.groupby('day')['frac_not_docked'].mean().reset_index()

# --- Rainfall data ---
rain = pd.read_csv('daily_rainfall.csv')   # replace with actual filename
rain['date'] = pd.to_datetime(rain['date'], format='%d-%b-%Y')  # parse "01-jan-1948"
rain_jan = rain[rain['date'].dt.month == 1]

# Align years if needed (your rainfall dataset is 1948, bike data is 2022)
# For demonstration, just match on day/month ignoring year:
rain_jan['day'] = rain_jan['date'].dt.day
daily_summary['day_num'] = pd.to_datetime(daily_summary['day']).dt.day

# Merge on day of month
merged = pd.merge(daily_summary, rain_jan, left_on='day_num', right_on='day')

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bike usage (bar chart)
ax1.bar(merged['day'], merged['frac_not_docked'], color='tab:blue', alpha=0.6)
ax1.set_xlabel('Day in January')
ax1.set_ylabel('Fraction of Bikes Not Docked (in use)', color='tab:blue')

# Rainfall (line chart on secondary axis)
ax2 = ax1.twinx()
ax2.plot(merged['day'], merged['rain'], color='tab:green', marker='o')
ax2.set_ylabel('Rainfall (mm)', color='tab:green')

plt.title(f'Bike Usage vs Rainfall (Station {station_id}, January {year})')
fig.autofmt_xdate()
fig.tight_layout()

path = f'graphs/station_{station_id}_daily_january_{year}_with_rain.png'
fig.savefig(path)
plt.close(fig)

print(f"Saved January bike vs rainfall plot: {path}")