from bigdata_prep import create_sessions, create_machines, create_merged

sessions = create_sessions()
machines = create_machines()
merged = create_merged(sessions, machines)

# Group by date and machineid
aggregated = merged.groupby(["date", "machineid"]).agg({
    "coinin": "sum",
    "coinout": "sum",
    "sessionlength": "sum",
    "numgames": "sum",
    "name": "first",
    "asset": "first",
    "removedate": "first",
    "multidenom": "first",
    "multigame": "first",
    "clientid": "first",
    "holdpct": "first",
    "paybackpct": "first",
    "x": "first",
    "y": "first"
}).reset_index()

# Add numsessions column (count occurrences of each date and machineid)
aggregated["numsessions"] = merged.groupby(["date", "machineid"]).size().values

# Calculate new avgbet
aggregated["avgbet"] = aggregated["coinin"] / aggregated["numgames"]

# Save the new DataFrame to a CSV file
output_path = "Data/Merged_By_Date_MachineID_Active.csv"
aggregated.to_csv(output_path, index=False)

print(f"Aggregated data saved to {output_path}")
