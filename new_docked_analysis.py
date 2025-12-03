# Histogram showing percentage of bikes over the course of the Michaelmas term.

import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combined_cleaned.csv')

station_id = 98   # replace with actual station ID
year = 2023
station_data = df[(df['STATION ID'] == station_id) & (df['YEAR'] == year)]
station_data['TIME'] = pd.to_datetime(station_data['TIME'])
station_data['week'] = station_data['TIME'].dt.isocalendar().week
station_data["frac_not_docked"] = (station_data["BIKE_STANDS"] - station_data["AVAILABLE_BIKES"]) / station_data["BIKE_STANDS"]
weekly_summary = station_data.groupby('week')['frac_not_docked'].mean()

###

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(weekly_summary.index, weekly_summary.values, width=0.8)

ax.set_xlabel('Week of Year')
ax.set_ylabel('Fraction of Bikes Not Docked (in use)')
ax.set_title(f'Weekly Bike Usage for Station {station_id} in {year}')
fig.tight_layout()

path = f'graphs/station_{station_id}_weekly_{year}.png'
fig.savefig(path)
plt.close(fig)

print(f"Saved weekly histogram: {path}")