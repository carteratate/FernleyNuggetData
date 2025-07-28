import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === STEP 1: Parse args and load full dataset ===
parser = argparse.ArgumentParser(description="Extend features for next month")
parser.add_argument("--features", default="Data/features.csv", help="Input features CSV")
parser.add_argument("--output", default="Data/future_month_testdata.csv", help="Output future layout CSV")
args = parser.parse_args()

df = pd.read_csv(args.features)

# === STEP 2: Find the layout for July 2025 on a Thursday ===
# Filter rows where is_2025 == 1, is_July == 1, and is_Thursday == 1
specific_date_mask = df[
    (df["is_2025"] == 1) & (df["is_July"] == 1) & (df["is_thursday"] == 1)
]

# Drop duplicates to get machine-level layout (assumes one row per machine per day)
latest_layout = specific_date_mask.drop_duplicates(subset=["x", "y"])

# === STEP 3: Build next month's dates ===
start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
future_dates = [start_date + timedelta(days=i) for i in range(31)]

# === STEP 4: Generate rows for each day ===
extended_rows = []

for day in future_dates:
    for _, row in latest_layout.iterrows():
        new_row = row.copy()

        # Reset coinin to 0 (we'll predict it later)
        new_row["coinin"] = 0

        # Date-based features
        dow = day.weekday()  # Monday=0
        month = day.month
        year_val = day.year

        # One-hot encodings
        new_row["day_of_week"] = dow
        for i, dayname in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            new_row[f"is_{dayname}"] = 1 if dow == i else 0

        new_row["is_weekend"] = 1 if dow >= 5 else 0

        for i, monthname in enumerate([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ], start=1):
            new_row[f"is_{monthname}"] = 1 if month == i else 0

        new_row["month"] = month

        for y in range(2020, 2026):
            new_row[f"is_{y}"] = 1 if year_val == y else 0

        new_row["year"] = year_val - 2020  # Year encoded as 0 for 2020, 1 for 2021, etc.

        extended_rows.append(new_row)

# === STEP 5: Save the full feature set ===
future_df = pd.DataFrame(extended_rows)  # Convert extended rows to a DataFrame
future_df.to_csv(args.output, index=False)  # Save full feature set
print(f"Saved future layout to '{args.output}'")


# === STEP 6: Generate machine layout ===
# Dynamically determine theme columns based on their position in the DataFrame
theme_columns = future_df.iloc[:, 50:184].columns.tolist()  

# Extract x, y, and the theme where is_theme is true
machine_layout = future_df.loc[:, ["x", "y"] + theme_columns]

# Find the theme where is_theme is true for each machine
machine_layout["theme"] = machine_layout[theme_columns].idxmax(axis=1).str.replace("is_", "")

# Drop duplicates to ensure only one machine per x, y location
machine_layout = machine_layout.drop_duplicates(subset=["x", "y"])

# Drop the theme flag columns to keep only x, y, and theme
machine_layout = machine_layout.loc[:, ["x", "y", "theme"]]

# Save the machine layout to a separate CSV file
machine_layout.to_csv("Data/future_month_layout.csv", index=False)
