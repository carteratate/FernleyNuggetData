import argparse
import pandas as pd
from utils import create_sessions, create_machines, create_merged


parser = argparse.ArgumentParser(description="Clean and merge CT data")
parser.add_argument("--sessions", default="Data/assetmeters 07-25-2025.csv", help="Sessions CSV")
parser.add_argument("--machines", default="Data/TTgames00 07-25-2025.csv", help="Machines CSV")
parser.add_argument("--output", default="Data/ctdata_amtt_merge.csv", help="Output merged CSV")
args = parser.parse_args()

session_map = {
    1: "asset",
    5: "vendor",
    6: "theme",
    12: "isprogressive",
    13: "holdpct",
    14: "denom",
    17: "x",
    18: "y",
    21: "time",
    24: "coinin",
    25: "coinout",
    28: "gamesplayed",
}
sessions = create_sessions(args.sessions, rename_map=session_map)
sessions = sessions[[
    "asset", "vendor", "theme", "isprogressive", "holdpct", "denom",
    "x", "y", "time", "coinin", "coinout", "gamesplayed",
]]

machine_map = {
    13: "serial",
    24: "multidenom",
    27: "maxbet",
    52: "x_coord",
    53: "y_coord",
}
machines = create_machines(args.machines, rename_map=machine_map)
machines = machines[["serial", "multidenom", "maxbet", "x_coord", "y_coord"]]

merged = create_merged(sessions, machines, "asset", "serial")
merged.to_csv(args.output, index=False)
