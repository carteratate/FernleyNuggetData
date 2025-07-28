import argparse
import pandas as pd
from utils import create_sessions, create_machines, create_merged

parser = argparse.ArgumentParser(description="Prepare bigdata sessions and machines")
parser.add_argument("--sessions", default="Data/BigData7_24_25.csv", help="Sessions CSV")
parser.add_argument("--machines", default="Data/SlotsOnFloor.csv", help="Machines CSV")
parser.add_argument("--output", default="Data/bigdata_merge_active.csv", help="Merged output CSV")
args = parser.parse_args()

session_map = {
    2: "machine",
    3: "plxid",
    5: "date",
    6: "time",
    7: "sessionlength",
    8: "avgbet",
    9: "coinin",
    10: "coinout",
    12: "holdpercent",
}

sessions = create_sessions(args.sessions, rename_map=session_map)
sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 512100000
sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000
sessions = sessions.dropna(subset=["playerid", "machineid"])
print(sessions["machineid"].isnull().sum())  # Should be 0
sessions["numgames"] = sessions["coinin"] / sessions["avgbet"]
sessions = sessions[[
    "playerid", "machineid", "date", "time", "sessionlength", "avgbet", "coinin", "coinout", "numgames", "holdpercent",
]]

machine_map = {
    0: "id",
    2: "name",
    6: "asset",
    9: "removedate",
    12: "multidenom",
    13: "multigame",
    17: "raw_clientid",
    18: "holdpct",
    19: "paybackpct",
    32: "x",
    33: "y",
}

machines = create_machines(args.machines, rename_map=machine_map)
machines["clientid"] = pd.to_numeric(machines["raw_clientid"], errors="coerce") - 512100000000
machines = machines.dropna(subset=["clientid"])
machines = machines.dropna(subset=["id"])
machines = machines[machines["removedate"].isnull()]
print(machines["clientid"].duplicated().sum())
print(machines["id"].duplicated().sum())
machines = machines[[
    "id", "name", "asset", "removedate", "multidenom", "multigame", "clientid", "holdpct", "paybackpct", "x", "y",
]]

merged = create_merged(sessions, machines, "machineid", "id")
print(f"Unmatched rows: {len(merged[merged['id'].isnull()])}")
merged.to_csv(args.output, index=False)

