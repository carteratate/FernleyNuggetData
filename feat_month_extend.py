import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main() -> None:
    """Extend the feature matrix to the next month."""

    # === STEP 1: Load full dataset ===
    df = pd.read_csv("Data/features.csv")

    # === STEP 2: Find most recent date layout ===
    # Heuristic: take the most frequent combo of year/month/day_of_week
    most_recent_mask = df[
        (df["is_2025"] == 1) & (df["month"] == df[df["is_2025"] == 1]["month"].max())
    ]

    # Drop duplicates to get machine-level layout (assumes one row per machine per day)
    latest_layout = most_recent_mask.drop_duplicates(subset=["x", "y"])

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
    
    # === STEP 5: Assemble into new DataFrame ===
    future_df = pd.DataFrame(extended_rows)
    
    # === STEP 6: Save output ===
    future_df.to_csv("Data/future_month_layout.csv", index=False)
    print("Saved future layout to 'Data/future_month_layout.csv'")

if __name__ == "__main__":
    main()
