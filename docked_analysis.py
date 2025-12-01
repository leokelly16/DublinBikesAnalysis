import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combined_cleaned.csv')
df['TIME'] = pd.to_datetime(df['TIME'])
df['month'] = df['TIME'].dt.month

df['frac_not_docked'] = (df['BIKE_STANDS'] - df['AVAILABLE_BIKES']) / df['BIKE_STANDS']

term_months = list(range(9, 13)) + list(range(1, 6))

os.makedirs('graphs', exist_ok=True)

for year in [2022, 2023]:
    print(f"Processing year {year}...")

    yearly = df[df['YEAR'] == year]
    summaries = []

    for station, name in yearly.groupby('STATION ID'):
        term = name[name['month'].isin(term_months)]['frac_not_docked'].mean()
        out = name[~name['month'].isin(term_months)]['frac_not_docked'].mean()
        summaries.append((station, term, out))

    summ_df = pd.DataFrame(summaries, columns=['station', 'term', 'out'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summ_df))

    ax.bar([i - 0.2 for i in x], summ_df['term'], width=0.4, label='In Term Time')
    ax.bar([i + 0.2 for i in x], summ_df['out'], width=0.4, label='Out of Term Time')

    ax.set_xticks(x)
    ax.set_xticklabels(summ_df['station'])
    ax.set_ylabel('Fraction of Bikes Not Docked (in use)')
    ax.set_title(f'Bike Not Docked (in use) by Station {year}')
    ax.legend()
    fig.tight_layout()

    path = f'graphs/bike_not_docked_{year}.png'
    fig.savefig(path)
    plt.close(fig)

    print(f"Saved graph: {path}")

print("Done.")
