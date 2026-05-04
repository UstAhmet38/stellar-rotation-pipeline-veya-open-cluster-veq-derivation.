"""
01_isochrone_matching.py
------------------------
Matches observed Gaia DR3 photometry to PARSEC isochrone models using
minimum Euclidean distance in the color-magnitude diagram (CMD).

Derives stellar physical parameters (Teff, Luminosity, Radius) for each
matched star via isochrone interpolation and the Stefan-Boltzmann law.

Reference: Usta et al. (2025), PASP - Section 2.3
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# SETTINGS
# =============================================================================

# Matching threshold in magnitudes (Euclidean distance in CMD space).
# Stars closer than this value to the isochrone are considered matched.
THRESHOLD_VAL = 0.15

# Base directory: set this to the folder containing your cluster CSV and
# isochrone TXT files. Output will be saved to a subfolder named "output".
BASE_DIR = "data"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Gaia DR3 extinction coefficients (Cardelli law, standard values)
R_G  = 0.836
R_BP = 1.083
R_RP = 0.634

# Sun's effective temperature (K) - used in Stefan-Boltzmann radius derivation
T_SUN = 5777.0

# =============================================================================
# CLUSTER LIST
# Edit this list to add or remove clusters.
# Each entry requires:
#   name      : cluster identifier (used in output filenames)
#   dist_mod  : distance modulus (mag)
#   Av        : V-band extinction
#   obs_file  : filename of the Gaia DR3 member catalog (CSV)
#   iso_file  : filename of the PARSEC isochrone file (TXT/space-separated)
# =============================================================================
clusters = [
    {
        "name": "BLANCO1",
        "dist_mod": 6.95, "Av": 0.05,
        "obs_file": "BLANCO1.csv",
        "iso_file": "BLANCO1_ISO.txt"
    },
    {
        "name": "HYADES",
        "dist_mod": 3.35, "Av": 0.01,
        "obs_file": "HYADES.csv",
        "iso_file": "HYADES_ISO.txt"
    },
    {
        "name": "NGC2264",
        "dist_mod": 9.35, "Av": 0.22,
        "obs_file": "NGC2264.csv",
        "iso_file": "NGC2264_ISO.txt"
    },
    {
        "name": "NGC2281",
        "dist_mod": 8.75, "Av": 0.25,
        "obs_file": "NGC2281.csv",
        "iso_file": "NGC2281_ISO.txt"
    },
    {
        "name": "NGC3532",
        "dist_mod": 8.45, "Av": 0.12,
        "obs_file": "NGC3532.csv",
        "iso_file": "NGC3532_ISO.txt"
    },
    {
        "name": "NGC3766",
        "dist_mod": 11.30, "Av": 0.55,
        "obs_file": "NGC3766.csv",
        "iso_file": "NGC3766_ISO.txt"
    },
    {
        "name": "NGC6811",
        "dist_mod": 10.30, "Av": 0.35,
        "obs_file": "NGC6811.csv",
        "iso_file": "NGC6811_ISO.txt"
    },
    {
        "name": "NGC6819",
        "dist_mod": 11.85, "Av": 0.40,
        "obs_file": "NGC6819.csv",
        "iso_file": "NGC6819_ISO.txt"
    },
    {
        "name": "PLEIADES",
        "dist_mod": 5.65, "Av": 0.12,
        "obs_file": "PLEIADES.csv",
        "iso_file": "PLEIADES_ISO.txt"
    },
    {
        "name": "PRAESEPE",
        "dist_mod": 6.35, "Av": 0.09,
        "obs_file": "PRAESEPE.csv",
        "iso_file": "PRAESEPE_ISO.txt"
    },
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_isochrone(path, dist_mod, av):
    """
    Read a PARSEC isochrone file and convert absolute magnitudes to
    apparent magnitudes using the distance modulus and extinction.

    Parameters
    ----------
    path     : str   - full path to the isochrone file
    dist_mod : float - distance modulus (mag)
    av       : float - V-band extinction

    Returns
    -------
    pd.DataFrame with columns 'G_App' and 'Color_App', or None on failure.
    """
    if not os.path.exists(path):
        print(f"   ERROR: Isochrone file not found: {path}")
        return None

    try:
        # Locate the header row dynamically (handles both PARSEC column naming schemes)
        header_row = 0
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if "Gmag" in line or "G_fSBmag" in line:
                    header_row = i
                    break

        df = pd.read_csv(path, sep=r"\s+", header=header_row, comment="#")

        # Standardize column names across PARSEC versions
        g_col  = "Gmag"       if "Gmag"       in df.columns else "G_fSBmag"
        bp_col = "G_BPmag"    if "G_BPmag"    in df.columns else "G_BP_fSBmag"
        rp_col = "G_RPmag"    if "G_RPmag"    in df.columns else "G_RP_fSBmag"

        # Compute extinction in Gaia bands
        A_G      = R_G * av
        E_BP_RP  = (R_BP - R_RP) * av

        # Shift from absolute to apparent (observed) frame
        df["G_App"]     = df[g_col]                    + dist_mod + A_G
        df["Color_App"] = (df[bp_col] - df[rp_col])   + E_BP_RP

        return df

    except Exception as exc:
        print(f"   ERROR loading isochrone: {exc}")
        return None


def match_stars_to_isochrone(df_obs, df_iso, obs_col_name, threshold):
    """
    Match each observed star to its nearest isochrone point in CMD space
    using the Euclidean distance metric (Eq. 1 in Usta et al. 2025).

    Parameters
    ----------
    df_obs       : pd.DataFrame - observed member catalog
    df_iso       : pd.DataFrame - isochrone in apparent-magnitude frame
    obs_col_name : str          - name of the BP-RP color column in df_obs
    threshold    : float        - maximum allowed CMD distance for a match

    Returns
    -------
    list of dicts, one per matched star, containing observed + isochrone data
    and derived physical parameters.
    """
    obs_col_arr = df_obs[obs_col_name].values
    obs_mag_arr = df_obs["G"].values
    iso_col_arr = df_iso["Color_App"].values
    iso_mag_arr = df_iso["G_App"].values

    matched = []

    for i in range(len(df_obs)):
        c_obs = obs_col_arr[i]
        m_obs = obs_mag_arr[i]

        if np.isnan(c_obs) or np.isnan(m_obs):
            continue

        # Euclidean distance to all isochrone points in CMD space
        distances = np.sqrt((iso_col_arr - c_obs)**2 + (iso_mag_arr - m_obs)**2)
        min_dist  = distances.min()
        min_idx   = distances.argmin()

        if min_dist >= threshold:
            continue

        star = df_obs.iloc[i].to_dict()
        iso_row = df_iso.iloc[min_idx].to_dict()

        # ----------------------------------------------------------------
        # Physical parameter derivation (Section 2.3, Usta et al. 2025)
        # ----------------------------------------------------------------

        # Luminosity: L/L_sun = 10^(logL)
        L_lum = None
        if "logL" in iso_row:
            L_lum = 10.0 ** iso_row["logL"]
            star["Calc_Luminosity"] = L_lum

        # Effective temperature: Teff = 10^(logTe)  [K]
        T_eff = None
        if "logTe" in iso_row:
            T_eff = 10.0 ** iso_row["logTe"]
            star["Calc_Teff"] = T_eff

        # Radius via Stefan-Boltzmann law (Eq. 2):  R/R_sun = sqrt(L) * (T_sun/T)^2
        if L_lum is not None and T_eff is not None:
            star["Calc_Radius"] = np.sqrt(L_lum) * (T_SUN / T_eff) ** 2
        else:
            star["Calc_Radius"] = np.nan

        # Store isochrone columns with prefix to avoid name collisions
        for k, v in iso_row.items():
            if k not in ("G_App", "Color_App"):
                star[f"ISO_{k}"] = v

        star["Match_Distance"] = min_dist
        matched.append(star)

    return matched


# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"Starting isochrone matching for {len(clusters)} clusters ...")
print("=" * 60)

for cluster in clusters:
    name     = cluster["name"]
    obs_path = os.path.join(BASE_DIR, cluster["obs_file"])
    iso_path = os.path.join(BASE_DIR, cluster["iso_file"])

    print(f"\n>>> Cluster: {name}")
    print(f"    dist_mod={cluster['dist_mod']}, Av={cluster['Av']}")

    # --- Load observed catalog ---
    if not os.path.exists(obs_path):
        print(f"    SKIPPED: observation file not found ({obs_path})")
        continue

    try:
        df_obs = pd.read_csv(obs_path)
        df_obs.columns = df_obs.columns.str.strip()

        # Identify the BP-RP color column
        if "bp_rp" in df_obs.columns:
            obs_col_name = "bp_rp"
        elif "BP_RP" in df_obs.columns:
            obs_col_name = "BP_RP"
        else:
            print("    ERROR: No BP-RP color column found in observation file.")
            continue

    except Exception as exc:
        print(f"    ERROR loading observation file: {exc}")
        continue

    # --- Load isochrone ---
    df_iso = load_isochrone(iso_path, cluster["dist_mod"], cluster["Av"])
    if df_iso is None:
        continue

    # --- Match ---
    print(f"    Matching stars (threshold = {THRESHOLD_VAL} mag) ...")
    matched = match_stars_to_isochrone(df_obs, df_iso, obs_col_name, THRESHOLD_VAL)

    # --- Save ---
    if not matched:
        print("    WARNING: No stars matched. Consider increasing THRESHOLD_VAL.")
        continue

    result_df = pd.DataFrame(matched)

    # Bring the most informative columns to the front
    priority = ["Calc_Radius", "Calc_Teff", "Calc_Luminosity", "ISO_Mass", "ISO_Mini"]
    priority = [c for c in priority if c in result_df.columns]
    rest     = [c for c in result_df.columns if c not in priority]
    result_df = result_df[priority + rest]

    out_path = os.path.join(OUTPUT_DIR, f"{name}_isochrone_match.csv")
    result_df.to_csv(out_path, index=False)
    print(f"    SUCCESS: {len(result_df)} stars matched -> {out_path}")

print("\n" + "=" * 60)
print("Done. Results saved to:", OUTPUT_DIR)
