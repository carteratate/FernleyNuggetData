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
def createSessions():
    sessions_path = "Data/PlayerOne.csv"

    sessions = pd.read_csv(sessions_path)

    sessions = clean_cols(sessions)

    sessions = sessions.rename(columns={
        sessions.columns[0]: "plxid",         # Adjust if needed
        sessions.columns[1]: "time",
        sessions.columns[2]: "sessionlength",
        sessions.columns[3]: "machine",
        sessions.columns[6]: "avgbet",
        sessions.columns[7]: "wager",
        sessions.columns[8]: "holdpercent"
    })

    # Prepare IDs (as before)
    sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 5121000000
    sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000

    # Drop any with missing keys
    sessions = sessions.dropna(subset=["playerid", "machineid"])

    # Derive numGames
    sessions["numgames"] = sessions["wager"] / sessions["avgbet"]

    # Keep only relevant columns
    sessions = sessions[[
        "playerid", "machineid", "time", "sessionlength", "avgbet", "wager", "numgames", "holdpercent"
    ]]
    return sessions


# Clean up machines DataFrame ------------------------
def createMachines():
    machines_path = "Data/SlotsOnFloor.csv"

    machines = pd.read_csv(machines_path)

    machines = clean_cols(machines)

    machines = machines.rename(columns={
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

    # Keep only desired columns
    machines = machines[[
        "name", "asset", "removedate", "multidenom", "multigame",
        "clientid", "holdpct", "paybackpct", "x", "y"
    ]]

    return machines

# Merge DataFrames -----------------------------------
def createMerged(sessions, machines):
    merged = sessions.merge(
        machines,
        left_on="machineid",
        right_on="clientid",
        how="left"
    )
    return merged



