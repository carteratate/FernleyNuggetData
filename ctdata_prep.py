import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Clean and merge CT data")
parser.add_argument("--sessions", default="Data/assetmeters 07-25-2025.csv", help="Sessions CSV")
parser.add_argument("--machines", default="Data/TTgames00 07-25-2025.csv", help="Machines CSV")
parser.add_argument("--output", default="Data/ctdata_amtt_merge.csv", help="Output merged CSV")
args = parser.parse_args()


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
    sessions_path = args.sessions

    sessions = pd.read_csv(sessions_path)

    sessions = clean_cols(sessions)

    sessions = sessions.rename(columns={
        sessions.columns[1]: "asset",         # Adjust if needed
        sessions.columns[5]: "vendor",
        sessions.columns[6]: "theme",
        sessions.columns[12]: "isprogressive",
        sessions.columns[13]: "holdpct",
        sessions.columns[14]: "denom",
        sessions.columns[17]: "x",
        sessions.columns[18]: "y",
        sessions.columns[21]: "time",
        sessions.columns[24]: "coinin",
        sessions.columns[25]: "coinout",
        sessions.columns[28]: "gamesplayed",
        
    })

    # Keep only relevant columns
    sessions = sessions[[
        "asset", "vendor", "theme", "isprogressive", "holdpct", "denom",
        "x", "y", "time", "coinin", "coinout", "gamesplayed",
    ]]
    return sessions


# Clean up machines DataFrame ------------------------
def create_machines():
    machines_path = args.machines

    machines = pd.read_csv(machines_path)

    machines = clean_cols(machines)

    machines = machines.rename(columns={
        machines.columns[13]: "serial", 
        machines.columns[24]: "multidenom",
        machines.columns[27]: "maxbet",
        machines.columns[52]: "x_coord",
        machines.columns[53]: "y_coord",
    })

    # Keep only desired columns
    machines = machines[[
        "serial", "multidenom", "maxbet", "x_coord", "y_coord"
    ]]

    return machines

# Merge DataFrames -----------------------------------
def create_merged(sessions, machines):
    # Ensure both columns are of the same type
    sessions["asset"] = sessions["asset"].astype(str).str.strip()
    machines["serial"] = machines["serial"].astype(str).str.strip()

    merged = sessions.merge(
        machines,
        left_on="asset",
        right_on="serial",
        how="left"
    )

    unmatched = merged[merged["serial"].isnull()]
    # print(f"Unmatched rows: {len(unmatched)}")

    return merged

sessions = create_sessions()
machines = create_machines()

merged = create_merged(sessions, machines)

merged.to_csv(args.output, index=False)