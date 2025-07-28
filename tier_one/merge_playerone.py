"""
Merge Data/PlayerOne1.xlsx … PlayerOne11.xlsx into Data/PlayerOne.csv
in natural numeric order (1, 2, 3 … 11).

Usage:
    python merge_playerone.py
"""

import argparse
from pathlib import Path
import re
import pandas as pd

parser = argparse.ArgumentParser(description="Merge PlayerOne Excel files")
parser.add_argument("--data-dir", default="Data", help="Directory containing PlayerOne Excel files")
parser.add_argument("--output", default="Data/PlayerOne.csv", help="Output CSV file")
parser.add_argument("--glob", default="PlayerOne*.xlsx", help="Glob pattern for Excel files")
parser.add_argument("--add-source-col", action="store_true", help="Add source file column", default=True)
args = parser.parse_args()

# ---------- configuration ----------
DATA_DIR = Path(args.data_dir)
GLOB_PATTERN = args.glob
OUTPUT_FILE = Path(args.output)
ADD_SOURCE_COL = args.add_source_col
# -----------------------------------

def numeric_key(path_obj: Path) -> int:
    """
    Extract the integer portion of PlayerOneN.xlsx for natural sorting.
    Example: PlayerOne10.xlsx -> 10
    """
    m = re.search(r"(\d+)", path_obj.stem)
    return int(m.group(1)) if m else -1  # puts unexpected filenames last

def main():
    # 1. Collect and sort files naturally (1,2,3…11)
    xlsx_files = sorted(DATA_DIR.glob(GLOB_PATTERN), key=numeric_key)

    if not xlsx_files:
        raise FileNotFoundError(f"No files match {GLOB_PATTERN} inside {DATA_DIR}")

    # 2. Read each file into a DataFrame
    frames = []
    for f in xlsx_files:
        df = pd.read_excel(f)  # reads first sheet by default
        if ADD_SOURCE_COL:
            df["source_file"] = f.stem
        frames.append(df)

    # 3. Concatenate vertically
    merged = pd.concat(frames, ignore_index=True)

    # 4. Save as CSV
    merged.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Merged {len(xlsx_files)} files "
        f"→ {merged.shape[0]} rows\nSaved to: {OUTPUT_FILE}"
    )

if __name__ == "__main__":
    main()

