import pandas as pd
import datetime

merged_path = "Data/merged_with_clusters.csv"

merged = pd.read_csv(merged_path)


# DATE BASED FEATURES
# Convert the time column to datetime and extract only the date
# merged["date"] = pd.to_datetime(merged["time"], utc=True).dt.tz_convert(None).dt.date
merged["date"] = pd.to_datetime(merged["time"], format="%Y-%m-%d %H:%M:%S.%f")
# merged["date"] = pd.to_datetime(merged["time"], errors="coerce")

# Drop the original time column if no longer needed
merged = merged.drop(columns=["time"])

# Create a column for day_of_week (Sunday = 0, Monday = 1, ..., Saturday = 6)
merged["day_of_week"] = merged["date"].apply(lambda x: datetime.date(x.year, x.month, x.day).weekday())

# Create columns for each day of the week (is_monday, is_sunday, etc.)
merged["is_sunday"] = (merged["day_of_week"] == 6).astype(int)  # Adjusted for Sunday = 6
merged["is_monday"] = (merged["day_of_week"] == 0).astype(int)
merged["is_tuesday"] = (merged["day_of_week"] == 1).astype(int)
merged["is_wednesday"] = (merged["day_of_week"] == 2).astype(int)
merged["is_thursday"] = (merged["day_of_week"] == 3).astype(int)
merged["is_friday"] = (merged["day_of_week"] == 4).astype(int)
merged["is_saturday"] = (merged["day_of_week"] == 5).astype(int)

# Create a column for is_weekend (1 for Saturday/Sunday, 0 for weekdays)
merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

# Create a column for month (January = 0, February = 1, ..., December = 11)
merged["month"] = merged["date"].apply(lambda x: x.month - 1)

# Create columns for each month (is_January, is_February, etc.)
merged["is_January"] = (merged["month"] == 0).astype(int)
merged["is_February"] = (merged["month"] == 1).astype(int)
merged["is_March"] = (merged["month"] == 2).astype(int)
merged["is_April"] = (merged["month"] == 3).astype(int)
merged["is_May"] = (merged["month"] == 4).astype(int)
merged["is_June"] = (merged["month"] == 5).astype(int)
merged["is_July"] = (merged["month"] == 6).astype(int)
merged["is_August"] = (merged["month"] == 7).astype(int)
merged["is_September"] = (merged["month"] == 8).astype(int)
merged["is_October"] = (merged["month"] == 9).astype(int)
merged["is_November"] = (merged["month"] == 10).astype(int)
merged["is_December"] = (merged["month"] == 11).astype(int)

# Create a column for year (2020 = 0, 2021 = 1, ..., 2025 = 5)
merged["year"] = merged["date"].apply(lambda x: x.year - 2020)

# Create columns for each year (is_2020, is_2021, etc.)
merged["is_2020"] = (merged["year"] == 0).astype(int)
merged["is_2021"] = (merged["year"] == 1).astype(int)
merged["is_2022"] = (merged["year"] == 2).astype(int)
merged["is_2023"] = (merged["year"] == 3).astype(int)
merged["is_2024"] = (merged["year"] == 4).astype(int)
merged["is_2025"] = (merged["year"] == 5).astype(int)




#EXTRA SPATIAL BASED FEATURES (Clusters and X Y already added)

# Create a feature called bar_slot based on x and y columns
merged["bar_slot"] = ((merged["y"] < 20) & (merged["x"] > 30)).astype(int)

# Create a feature called near_bar based on x and y columns
merged["near_bar"] = (
    ((merged["x"] > 15) & (merged["x"] < 30) & (merged["y"] < 20)) | 
    ((merged["x"] > 30) & (merged["y"] > 20) & (merged["y"] < 40))
).astype(int)

# Create a feature called near_main_door based on x and y columns
merged["near_main_door"] = (
    ((merged["x"] < 12) & (merged["y"] < 18)) | 
    ((merged["y"] < 12) & (merged["x"] < 25))
).astype(int)

# Create a feature called near_back_door based on x and y columns
merged["near_back_door"] = (
    ((merged["x"] > 10) & (merged["x"] < 50) & (merged["y"] > 70))
).astype(int)

# Create binary features for each cluster (is_cluster0 to is_cluster12)
for cluster_id in range(13):  # Iterate from 0 to 12
    merged[f"is_cluster{cluster_id}"] = (merged["cluster_id"] == cluster_id).astype(int)








# GAME BASED FEATURES

# Clean the theme column to remove odd spacing and ensure consistent formatting
merged["theme"] = merged["theme"].str.strip()

# Remove rows where the theme is "variance" or "cash promo"
merged = merged.loc[~merged["theme"].isin(["Variance", "CASH PROMO"])]

# Get a list of unique themes (no repetitions)
unique_themes = merged["theme"].unique()

# Create binary features for each theme
theme_features = pd.DataFrame(
    {f"is_{theme.strip().replace(' ', '_').replace('__', '_').lower().rstrip('_')}": 
     (merged["theme"] == theme).astype(int)
     for theme in unique_themes}
)

# Add the theme features to the original DataFrame
merged = pd.concat([merged, theme_features], axis=1)


# Group by theme and calculate the average coinin per day
theme_avg_coinin = merged.groupby("theme")["coinin"].mean().round(6).to_dict()
# Create a new column called theme_strength based on the theme_avg_coinin dictionary
merged["theme_strength"] = merged["theme"].map(theme_avg_coinin)


# Group by theme and calculate the sum and count of coinin
theme_stats = merged.groupby("theme")["coinin"].agg(["sum", "count"]).rename(columns={"sum": "total_coinin", "count": "count_coinin"})

# Merge the stats back into the original DataFrame
merged = merged.merge(theme_stats, on="theme", how="left")

# Calculate theme_strength excluding the current row's coinin to 
# This uses out-of-fold aggregation to avoid data leakage
merged["theme_strength"] = ((merged["total_coinin"] - merged["coinin"]) / (merged["count_coinin"] - 1)).round(6)


# Clean the multidenom column to remove odd spacing and ensure consistent formatting
merged['multidenom'] = merged['multidenom'].astype(str).str.strip().str.lower()
# Map 'n' to 0, 'y' to 1, and keep all numeric 0s as 0 in multidenom_flag
merged['multidenom_flag'] = merged['multidenom'].apply(lambda x: 0 if x == 'n' or x == '0' else 1)






# # Fix weird max bet values
# # Divide individual values greater than 100 by 100 in the maxbet column
# merged["maxbet"] = merged["maxbet"].apply(lambda x: x / 100 if x > 100 else x)

# Remove unnecessary columns
merged = merged.drop(columns=["date", "asset", "theme", "multidenom", "maxbet", "isprogressive", "vendor", "serial", "x_coord", 
                              "y_coord", "coinout", "total_coinin", "count_coinin"])



# Save the updated DataFrame back to a CSV file
merged.to_csv("Data/features.csv", index=False)