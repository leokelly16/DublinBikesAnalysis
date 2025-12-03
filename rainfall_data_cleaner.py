import pandas as pd


rain = pd.read_csv('uncleaned_csv/daily_rainfall.csv')

# Keep only the date and rain columns
rain = rain[['date', 'rain']]

# Parse the date column
rain['date'] = pd.to_datetime(rain['date'], format='%d-%b-%Y')

# Convert rainfall to numeric
rain['rain'] = pd.to_numeric(rain['rain'])

# Filter for years 2022 and 2023
rain_filtered = rain[rain['date'].dt.year.isin([2022, 2023])]

# Save filtered file
rain_filtered.to_csv('rainfall_2022_2023.csv', index=False)

print("Saved rainfall data for 2022-2023 to rainfall_2022_2023.csv")
print(rain_filtered.head())