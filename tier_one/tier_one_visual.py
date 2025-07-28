import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from utils import create_sessions, create_machines, create_merged

SESSION_PATH = "Data/PlayerOne.csv"
MACHINE_PATH = "Data/SlotsOnFloor.csv"

session_map = {
    0: "plxid",
    1: "time",
    2: "sessionlength",
    3: "machine",
    6: "avgbet",
    7: "wager",
    8: "holdpercent",
}

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

def drawTotalCoinInHeatMap():
    sessions = create_sessions(SESSION_PATH, rename_map=session_map)
    sessions["playerid"] = pd.to_numeric(sessions["plxid"], errors="coerce") - 5121000000
    sessions["machineid"] = pd.to_numeric(sessions["machine"], errors="coerce") - 512100000000
    sessions = sessions.dropna(subset=["playerid", "machineid"])
    sessions["numgames"] = sessions["wager"] / sessions["avgbet"]
    sessions = sessions[[
        "playerid", "machineid", "time", "sessionlength", "avgbet", "wager", "numgames", "holdpercent",
    ]]
    machines = create_machines(MACHINE_PATH, rename_map=machine_map)
    machines["clientid"] = pd.to_numeric(machines["raw_clientid"], errors="coerce") - 512100000000
    machines = machines.dropna(subset=["clientid"])
    machines = machines[[
        "name", "asset", "removedate", "multidenom", "multigame", "clientid", "holdpct", "paybackpct", "x", "y",
    ]]
    merged = create_merged(sessions, machines, "machineid", "clientid")

    # 1. Aggregate total wager per (x, y)
    agg = merged.groupby(['x', 'y']).agg(total_wager=('wager', 'sum')).reset_index()

    # 2. Pivot so x = rows, y = columns
    pivot = agg.pivot(index='x', columns='y', values='total_wager')

    plt.figure(figsize=(12, 10))
    plt.imshow(
        pivot,
        origin='upper',      # (0,0) top left
        cmap='coolwarm',     # or your favorite!
        aspect='equal',
        vmin=0,
        vmax=np.nanpercentile(pivot.values, 85)
    )
    plt.colorbar(label='Total Coin In ($)')
    plt.title("Fernley Nugget Casino Floor Heatmap (Tier 1)")
    plt.xlabel('Y coordinate (right)')
    plt.ylabel('X coordinate (down)')

    # Ticks: columns are y, rows are x
    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=pivot.columns
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index
    )

    plt.tight_layout()
    plt.show()

drawTotalCoinInHeatMap()

