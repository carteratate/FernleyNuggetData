import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Clean up column names --------------------------------
def clean_cols(df):
    df.columns = (
        df.columns
        .str.lower()           # all lower-case
        .str.strip()           # remove leading/trailing spaces
        .str.replace(" ", "_") # optional: turn inner spaces to underscores
    )
    return df

# Clean up sessions DataFrame ------------------------
def create_sessions():
    sessions_path = "Data/BigData7_24_25.csv"

    sessions = pd.read_csv(sessions_path)

    sessions = clean_cols(sessions)

    sessions = sessions.rename(columns={
        sessions.columns[2]: "machine",         # Adjust if needed
        sessions.columns[3]: "plxid",
        sessions.columns[5]: "date",
        sessions.columns[6]: "time",
        sessions.columns[7]: "sessionlength",
        sessions.columns[8]: "avgbet",
        sessions.columns[9]: "coinin",
        sessions.columns[10]: "coinout",
        sessions.columns[12]: "holdpercent"
    })

    # Prepare IDs (as before)
    sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 5121000000
    sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000

    # Drop any with missing keys
    sessions = sessions.dropna(subset=["playerid", "machineid"])

    print(sessions["machineid"].isnull().sum())  # Should be 0

    # Derive numGames
    sessions["numgames"] = sessions["coinin"] / sessions["avgbet"]

    # Keep only relevant columns
    sessions = sessions[[
        "playerid", "machineid", "date", "time", "sessionlength", "avgbet", "coinin", "coinout", "numgames", "holdpercent"
    ]]
    return sessions


# Clean up machines DataFrame ------------------------
def create_machines():
    machines_path = "Data/SlotsOnFloor.csv"

    machines = pd.read_csv(machines_path)

    machines = clean_cols(machines)

    machines = machines.rename(columns={
        machines.columns[0]: "id", 
        machines.columns[2]: "name",
        machines.columns[6]: "asset",
        machines.columns[9]: "removedate",
        machines.columns[12]: "multidenom",
        machines.columns[13]: "multigame",
        machines.columns[17]: "raw_clientid",    
        machines.columns[18]: "holdpct",
        machines.columns[19]: "paybackpct",
        machines.columns[32]: "x",
        machines.columns[33]: "y"
    })

    # Prepare machine id for join
    machines["clientid"] = pd.to_numeric(machines["raw_clientid"], errors="coerce") - 512100000000
    machines = machines.dropna(subset=["clientid"])
    machines = machines.dropna(subset=["id"])

    # Drop machines with a remove date
    machines = machines[machines["removedate"].isnull()]
    
    print(machines["clientid"].duplicated().sum())  # Should be 0
    print(machines["id"].duplicated().sum())  # Should be 0

    # Keep only desired columns
    machines = machines[[
        "id", "name", "asset", "removedate", "multidenom", "multigame",
        "clientid", "holdpct", "paybackpct", "x", "y"
    ]]

    return machines

# Merge DataFrames -----------------------------------
def create_merged(sessions, machines):
    merged = sessions.merge(
        machines,
        left_on="machineid",
        right_on="id",
        how="left"
    )

    unmatched = merged[merged["id"].isnull()]
    print(f"Unmatched rows: {len(unmatched)}")

    return merged


def main() -> None:
    """Merge sessions and machine data and save to CSV."""

    sessions = create_sessions()
    machines = create_machines()

    merged = create_merged(sessions, machines)

    merged.to_csv("Data/bigdata_merge_active.csv", index=False)


if __name__ == "__main__":
    main()
