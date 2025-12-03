import pandas as pd
import matplotlib.pyplot as plt

# --- Bike usage data ---
bike = pd.read_csv('combined_cleaned.csv')

station_id = 98   # replace with your station ID
year = 2023       # year to analyze

# Filter for the chosen station and year
bike_data = bike[(bike['STATION ID'] == station_id) & (bike['YEAR'] == year)]
bike_data['TIME'] = pd.to_datetime(bike_data['TIME'])

# Restrict to January
bike_data = bike_data[bike_data['TIME'].dt.month == 1]

# Create a daily column
bike_data['day'] = bike_data['TIME'].dt.date

# Fraction of bikes docked (NOT in use)
bike_data['frac_docked'] = bike_data['AVAILABLE_BIKES'] / bike_data['BIKE_STANDS']

# Daily average usage
daily_bike = bike_data.groupby('day')['frac_docked'].mean().reset_index()
daily_bike['day'] = pd.to_datetime(daily_bike['day'])  # ensure datetime type

# --- Rainfall data ---
rain = pd.read_csv('rainfall_2022_2023.csv')  # already cleaned and filtered
rain['date'] = pd.to_datetime(rain['date'])

# Restrict to January
rain_jan = rain[rain['date'].dt.month == 1]

# --- Merge datasets on date ---
merged = pd.merge(daily_bike, rain_jan, left_on='day', right_on='date')

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bike usage (bar chart)
ax1.bar(merged['day'], merged['frac_docked'], color='tab:blue', alpha=0.6)
ax1.set_xlabel('Day in January')
ax1.set_ylabel('Fraction of Bikes Docked (NOT in use)', color='tab:blue')

# Rainfall (line chart on secondary axis)
ax2 = ax1.twinx()
ax2.plot(merged['day'], merged['rain'], color='tab:green', marker='o')
ax2.set_ylabel('Rainfall (mm)', color='tab:green')

plt.title(f'Bike Docking vs Rainfall (Station {station_id}, January {year})')
fig.autofmt_xdate()
fig.tight_layout()

# Save plot
path = f'graphs/station_{station_id}_bike_docked_vs_rainfall_january_{year}.png'
fig.savefig(path)
plt.close(fig)

print(f"Saved January bike docking vs rainfall plot: {path}")