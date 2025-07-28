import argparse
import pandas as pd
from utils import create_sessions, create_machines, create_merged

parser = argparse.ArgumentParser(description="Merge bigdata by date and machine")
parser.add_argument("--output", default="Data/Merged_By_Date_MachineID_Active.csv", help="Output merged CSV")
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

sessions = create_sessions("Data/BigData7_24_25.csv", rename_map=session_map)
sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 512100000
sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000
sessions = sessions.dropna(subset=["playerid", "machineid"])
sessions["numgames"] = sessions["coinin"] / sessions["avgbet"]
sessions = sessions[[
    "playerid", "machineid", "date", "time", "sessionlength", "avgbet", "coinin", "coinout", "numgames", "holdpercent",
]]

machines = create_machines("Data/SlotsOnFloor.csv", rename_map=machine_map)
machines["clientid"] = pd.to_numeric(machines["raw_clientid"], errors="coerce") - 512100000000
machines = machines.dropna(subset=["clientid"])
machines = machines.dropna(subset=["id"])
machines = machines[machines["removedate"].isnull()]
machines = machines[[
    "id", "name", "asset", "removedate", "multidenom", "multigame", "clientid", "holdpct", "paybackpct", "x", "y",
]]

merged = create_merged(sessions, machines, "machineid", "id")

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
    "y": "first",
}).reset_index()

aggregated["numsessions"] = merged.groupby(["date", "machineid"]).size().values
aggregated["avgbet"] = aggregated["coinin"] / aggregated["numgames"]
aggregated.to_csv(args.output, index=False)
print(f"Aggregated data saved to {args.output}")
