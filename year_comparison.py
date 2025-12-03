import os
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess
df = pd.read_csv('combined_cleaned.csv')
df['TIME'] = pd.to_datetime(df['TIME'])
df['frac_not_docked'] = (df['BIKE_STANDS'] - df['AVAILABLE_BIKES']) / df['BIKE_STANDS']

# Compute mean usage per station per year
station_means = df.groupby(['YEAR', 'STATION ID'])['frac_not_docked'].mean().reset_index()

# Pivot so each station has columns for 2022 and 2023
pivot_means = station_means.pivot(index='STATION ID', columns='YEAR', values='frac_not_docked')

# Sort by 2023 usage (optional, makes chart easier to read)
pivot_means = pivot_means.sort_values(by=2023, ascending=False)

# Plot side-by-side bars
fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(pivot_means))

ax.bar([i - 0.2 for i in x], pivot_means[2022], width=0.4, label='2022')
ax.bar([i + 0.2 for i in x], pivot_means[2023], width=0.4, label='2023')

ax.set_xticks(x)
ax.set_xticklabels(pivot_means.index.astype(str), rotation=90)
ax.set_xlabel('Station ID')
ax.set_ylabel('Mean Fraction of Bikes Not Docked (in use)')
ax.set_title('Mean Bike Usage per Station: 2022 vs 2023')
ax.legend()
fig.tight_layout()

# Save
os.makedirs('graphs', exist_ok=True)
path = 'graphs/station_mean_usage_comparison.png'
fig.savefig(path)
plt.close(fig)

print(f"Saved comparison graph: {path}")