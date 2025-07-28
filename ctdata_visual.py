import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from utils import create_sessions, create_machines, create_merged

SESSION_PATH = "Data/assetmeters 07-25-2025.csv"
MACHINE_PATH = "Data/TTgames00 07-25-2025.csv"

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

machine_map = {
    13: "serial",
    24: "multidenom",
    27: "maxbet",
    52: "x_coord",
    53: "y_coord",
}

def drawTotalCoinInHeatMap():
    sessions = create_sessions(SESSION_PATH, rename_map=session_map)
    sessions = sessions[[
        "asset", "vendor", "theme", "isprogressive", "holdpct", "denom",
        "x", "y", "time", "coinin", "coinout", "gamesplayed",
    ]]
    machines = create_machines(MACHINE_PATH, rename_map=machine_map)
    machines = machines[["serial", "multidenom", "maxbet", "x_coord", "y_coord"]]
    merged = create_merged(sessions, machines, "asset", "serial")

    # 1. Aggregate total wager per (x, y)
    agg = merged.groupby(['x', 'y']).agg(total_wager=('coinin', 'sum')).reset_index()

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
    plt.title("Fernley Nugget Casino Floor Heatmap CT Total")
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

