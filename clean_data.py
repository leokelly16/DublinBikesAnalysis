import pandas as pd
import os
from pathlib import Path
import re

# Directories
INPUT_DIR = Path("uncleaned_csv")
OUTPUT_DIR = Path("cleaned_csv")
OUTPUT_DIR.mkdir(exist_ok=True)

# Combined dataset holder
combined_df = []

# Station IDs to keep
KEEP_STATIONS = {21, 22, 27, 32, 98}

# Regex to detect year and month in filenames (e.g. "dublinbike-historical-data-2022-07.csv")
YEAR_MONTH_PATTERN = re.compile(r"(\d{4})[-_](\d{2})")

csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in 'uncleaned_csv'.")
else:
    for file_path in csv_files:
        print(f"Processing: {file_path.name}")

        # Extract year and month from filename
        match = YEAR_MONTH_PATTERN.search(file_path.name)
        if match:
            year, month = match.groups()
        else:
            print(f"⚠️  Could not detect year/month in filename '{file_path.name}', skipping.")
            continue

        # Load CSV
        df = pd.read_csv(file_path)

        # Ensure required column exists
        if "STATION_ID" not in df.columns:
            print(f"⚠️  Skipping {file_path.name}: missing STATION ID column")
            continue

        # Filter by station IDs
        df_filtered = df[df["STATION ID"].isin(KEEP_STATIONS)]

        # Add year and month columns
        df_filtered["YEAR"] = int(year)
        df_filtered["MONTH"] = int(month)

        # Save individual cleaned CSV
        output_path = OUTPUT_DIR / file_path.name
        df_filtered.to_csv(output_path, index=False)
        print(f"✔️ Saved cleaned file to: {output_path}")

        # Collect for combined dataset
        combined_df.append(df_filtered)

    # Create and save combined dataset if anything was processed
    if combined_df:
        merged = pd.concat(combined_df, ignore_index=True)
        merged.to_csv("combined_cleaned.csv", index=False)
        print("📌 Combined dataset saved as 'combined_cleaned.csv'")
    else:
        print("⚠️ No cleaned data to combine.")

print("Done.")
