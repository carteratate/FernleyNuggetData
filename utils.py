import pandas as pd


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names.

    Lowercases, strips whitespace and replaces spaces with underscores.
    The input DataFrame is modified in-place and returned for convenience.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df


def create_sessions(path: str, rename_map: dict | None = None,
                    usecols: list[str] | None = None) -> pd.DataFrame:
    """Load and clean a sessions CSV.

    Parameters
    ----------
    path:
        CSV file to load.
    rename_map:
        Optional mapping of column indexes to desired names.
    usecols:
        Optional list of columns to retain after renaming.
    """
    df = pd.read_csv(path)
    df = clean_cols(df)
    if rename_map:
        df = df.rename(columns={df.columns[idx]: name for idx, name in rename_map.items()})
    if usecols:
        df = df[usecols]
    return df


def create_machines(path: str, rename_map: dict | None = None,
                     usecols: list[str] | None = None) -> pd.DataFrame:
    """Load and clean a machines CSV.

    Parameters
    ----------
    path:
        CSV file to load.
    rename_map:
        Optional mapping of column indexes to desired names.
    usecols:
        Optional list of columns to retain after renaming.
    """
    df = pd.read_csv(path)
    df = clean_cols(df)
    if rename_map:
        df = df.rename(columns={df.columns[idx]: name for idx, name in rename_map.items()})
    if usecols:
        df = df[usecols]
    return df


def create_merged(sessions: pd.DataFrame, machines: pd.DataFrame,
                   left_on: str, right_on: str) -> pd.DataFrame:
    """Merge sessions and machines on the specified keys."""
    sessions[left_on] = sessions[left_on].astype(str).str.strip()
    machines[right_on] = machines[right_on].astype(str).str.strip()
    merged = sessions.merge(machines, left_on=left_on, right_on=right_on, how="left")
    return merged


def assign_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spatial feature columns based on ``x``/``y`` coordinates and ``cluster_id``."""
    df["bar_slot"] = ((df["y"] < 20) & (df["x"] > 30)).astype(int)
    df["near_bar"] = (((df["x"] > 15) & (df["x"] < 30) & (df["y"] < 20)) |
                       ((df["x"] > 30) & (df["y"] > 20) & (df["y"] < 40))).astype(int)
    df["near_main_door"] = (((df["x"] < 12) & (df["y"] < 18)) |
                             ((df["y"] < 12) & (df["x"] < 25))).astype(int)
    df["near_back_door"] = (((df["x"] > 10) & (df["x"] < 50) & (df["y"] > 70))).astype(int)
    if "cluster_id" in df.columns:
        for cid in range(13):
            df[f"is_cluster{cid}"] = (df["cluster_id"] == cid).astype(int)
    return df
