import argparse
import pandas as pd
from utils import create_sessions, create_machines, create_merged

parser = argparse.ArgumentParser(description="Prepare Player One data")
parser.add_argument("--sessions", default="Data/PlayerOne.csv", help="Player sessions CSV")
parser.add_argument("--machines", default="Data/SlotsOnFloor.csv", help="Machines CSV")
args = parser.parse_args()

session_map = {
    0: "plxid",
    1: "time",
    2: "sessionlength",
    3: "machine",
    6: "avgbet",
    7: "wager",
    8: "holdpercent",
}

sessions = create_sessions(args.sessions, rename_map=session_map)
sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 5121000000
sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000
sessions = sessions.dropna(subset=["playerid", "machineid"])
sessions["numgames"] = sessions["wager"] / sessions["avgbet"]
sessions = sessions[[
    "playerid", "machineid", "time", "sessionlength", "avgbet", "wager", "numgames", "holdpercent"
]]

machine_map = {
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
machines = machines[[
    "name", "asset", "removedate", "multidenom", "multigame",
    "clientid", "holdpct", "paybackpct", "x", "y",
]]

merged = create_merged(sessions, machines, "machineid", "clientid")

