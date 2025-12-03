import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combined_cleaned.csv')

station_id = 21   # replace with actual station ID
year = 2023
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
daily_summary = station_data.groupby('day')['frac_not_docked'].mean()

### Plotting ###
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(daily_summary.index, daily_summary.values, width=0.8)

ax.set_xlabel('Day in January')
ax.set_ylabel('Fraction of Bikes Not Docked (in use)')
ax.set_title(f'Daily Bike Usage for Station {station_id} in January {year}')
fig.autofmt_xdate()  # Rotate date labels for readability
fig.tight_layout()

path = f'graphs/station_{station_id}_daily_january_{year}.png'
fig.savefig(path)
plt.close(fig)

print(f"Saved January daily histogram: {path}")