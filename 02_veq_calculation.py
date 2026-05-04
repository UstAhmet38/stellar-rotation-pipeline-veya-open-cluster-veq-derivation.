"""
02_veq_calculation.py
---------------------
Computes equatorial rotational velocities (Veq) for stars in the Silver Group
using stellar radii derived from isochrone matching and photometric rotation
periods (Prot).

Formula (Eq. 3, Usta et al. 2025):
    Veq = 2 * pi * R [km] / Prot [s]

Run this script after 01_isochrone_matching.py.

Reference: Usta et al. (2025), PASP - Section 2.3
"""

import pandas as pd
import numpy as np
import os
import glob

# =============================================================================
# SETTINGS
# =============================================================================

# Directory containing the output CSVs from 01_isochrone_matching.py
INPUT_DIR = os.path.join("data", "output")

# Physical constants
R_SUN_KM    = 695_700.0  # Solar radius in km
SEC_IN_DAY  = 86_400.0   # Seconds per day

# Accepted column names for the rotation period (days).
# The script will try each name in order and use the first one found.
PERIOD_COL_CANDIDATES = ["Prot", "prot", "Per", "Period", "P", "period"]

# =============================================================================
# FUNCTIONS
# =============================================================================

def find_period_column(df, filename):
    """
    Search for a rotation period column among known candidate names.

    Parameters
    ----------
    df       : pd.DataFrame
    filename : str - used only for warning messages

    Returns
    -------
    str or None - column name if found, else None
    """
    for col in PERIOD_COL_CANDIDATES:
        if col in df.columns:
            return col
    print(f"   WARNING: No rotation period column found in '{filename}'.")
    print(f"   Expected one of: {PERIOD_COL_CANDIDATES}")
    return None


def compute_veq(df, period_col):
    """
    Derive equatorial velocity from stellar radius and rotation period.

    Veq = 2 * pi * R_km / P_sec   (Eq. 3, Usta et al. 2025)

    Parameters
    ----------
    df         : pd.DataFrame - must contain 'Calc_Radius' and period_col
    period_col : str          - name of the Prot column

    Returns
    -------
    pd.Series of Veq values in km/s
    """
    radius_km  = df["Calc_Radius"] * R_SUN_KM       # R_sun -> km
    period_sec = df[period_col]    * SEC_IN_DAY      # days  -> seconds

    with np.errstate(divide="ignore", invalid="ignore"):
        veq = (2.0 * np.pi * radius_km) / period_sec

    veq = veq.replace([np.inf, -np.inf], np.nan)
    return veq.round(3)


# =============================================================================
# MAIN LOOP
# =============================================================================

files = glob.glob(os.path.join(INPUT_DIR, "*_isochrone_match.csv"))

if not files:
    print(f"No matched catalog files found in '{INPUT_DIR}'.")
    print("Run 01_isochrone_matching.py first.")
else:
    print(f"Computing Veq for {len(files)} file(s) ...")
    print("-" * 55)

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"> Processing: {filename}")

        try:
            df = pd.read_csv(filepath)

            # Check for required radius column
            if "Calc_Radius" not in df.columns:
                print(f"   ERROR: 'Calc_Radius' column missing. "
                      f"Re-run 01_isochrone_matching.py first.")
                print()
                continue

            # Locate rotation period column
            period_col = find_period_column(df, filename)
            if period_col is None:
                print()
                continue

            print(f"   Period column: '{period_col}'")

            # Compute and append Veq
            df["Calc_Veq"] = compute_veq(df, period_col)

            # Overwrite the file with the new column included
            df.to_csv(filepath, index=False)

            n_valid = df["Calc_Veq"].notna().sum()
            print(f"   SUCCESS: Veq computed for {n_valid} / {len(df)} stars.")
            print(f"   Saved  : {filepath}")

        except Exception as exc:
            print(f"   ERROR: {exc}")

        print()

    print("-" * 55)
    print("All done.")
