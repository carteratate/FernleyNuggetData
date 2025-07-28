from ctdata_prep import create_sessions, create_machines, create_merged
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

def drawTotalCoinInHeatMap():
    sessions = create_sessions()
    machines = create_machines()
    merged = create_merged(sessions, machines)

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


def main() -> None:
    """Generate and display the heatmap."""

    drawTotalCoinInHeatMap()


if __name__ == "__main__":
    main()
