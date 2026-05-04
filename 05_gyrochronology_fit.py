"""
05_gyrochronology_fit.py
------------------------
Fits cubic polynomial models to the Prot-color relation for each Silver Group
cluster using LOESS smoothing with iterative sigma-clipping, then plots the
composite gyrochronology diagram (Figure 7, Usta et al. 2025).

Two-step pipeline:
  Step 1 — Per-cluster LOESS fit + polynomial extraction
            Output: data/output/gyro_fit_parameters.csv  (Table 3)
                    data/output/per_cluster_plots/        (diagnostic figures)

  Step 2 — Composite gyrochronology diagram
            Output: data/output/gyrochronology_composite.png  (Figure 7)

Reference: Usta et al. (2025), PASP - Section 3.4 and Table 3
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================

# Folder containing per-cluster merged CSVs
# (output of 01_isochrone_matching.py + 02_veq_calculation.py)
INPUT_DIR  = os.path.join("data", "output")
OUTPUT_DIR = os.path.join("data", "output")

PLOT_DIR = os.path.join(OUTPUT_DIR, "per_cluster_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Main-sequence log g filter
LOG_G_MIN = 4.0
LOG_G_MAX = 5.0

# LOESS bandwidth (fraction of data used per local fit)
FRAC_VALUE = 0.30

# Sigma-clipping threshold
SIGMA_THRESHOLD = 2.0

# Minimum Prot cutoff per cluster (days).
# Set to None to apply no lower limit.
CLUSTER_SETTINGS = {
    "blanco1":  2.2,
    "ngc2264":  None,
    "ngc3532":  4.0,
    "ngc3766":  None,
    "praesepe": 5.0,
    "pleiades": 2.0,
    "ngc6819":  None,
    "ngc2281":  None,
    "ngc6811":  None,
    "hyades":   None,
}

# Cluster metadata for the composite plot (age in Myr, [Fe/H] in dex)
CLUSTER_DB = {
    "ngc2264":  {"label": "NGC 2264",  "age": 5,    "feh": -0.15},
    "ngc3766":  {"label": "NGC 3766",  "age": 20,   "feh": -0.47},
    "blanco1":  {"label": "Blanco 1",  "age": 115,  "feh":  0.003},
    "pleiades": {"label": "Pleiades",  "age": 130,  "feh": -0.05},
    "ngc3532":  {"label": "NGC 3532",  "age": 300,  "feh": -0.07},
    "ngc2281":  {"label": "NGC 2281",  "age": 435,  "feh":  0.00},
    "praesepe": {"label": "Praesepe",  "age": 750,  "feh":  0.14},
    "hyades":   {"label": "Hyades",    "age": 625,  "feh":  0.14},
    "ngc6811":  {"label": "NGC 6811",  "age": 1000, "feh":  0.00},
    "ngc6819":  {"label": "NGC 6819",  "age": 2500, "feh":  0.00},
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prot_limit(filename):
    """
    Return the minimum Prot cutoff for a given cluster filename.
    Returns (matched, limit): matched=True if the cluster is in CLUSTER_SETTINGS.
    """
    fname = filename.lower().replace(" ", "").replace("_", "")
    for key, limit in CLUSTER_SETTINGS.items():
        if key.replace("_", "") in fname:
            return True, limit
    return False, None


def find_col(df, candidates):
    """Case-insensitive column search. Returns first match or None."""
    for cand in candidates:
        match = next((c for c in df.columns if c.lower() == cand.lower()), None)
        if match:
            return match
    return None


def iterative_sigma_clip(x, y, n_iter=3, sigma=2.0, frac=0.3):
    """
    Remove outliers by iteratively fitting LOESS and rejecting points
    beyond `sigma` standard deviations from the fit residuals.

    Parameters
    ----------
    x, y    : 1-D arrays
    n_iter  : number of clipping iterations
    sigma   : rejection threshold in units of residual std
    frac    : LOESS bandwidth fraction

    Returns
    -------
    Boolean mask (True = keep).
    """
    mask = np.ones(len(x), dtype=bool)
    for _ in range(n_iter):
        x_c, y_c = x[mask], y[mask]
        if len(x_c) < 10:
            break
        idx = np.argsort(x_c)
        try:
            lo = sm.nonparametric.lowess(y_c[idx], x_c[idx], frac=frac, it=3)
            xu, ui = np.unique(lo[:, 0], return_index=True)
            y_pred = np.interp(x_c, xu, lo[ui, 1])
            resid  = y_c - y_pred
            bad    = np.abs(resid) > sigma * np.std(resid)
            good_idx = np.where(mask)[0]
            mask[good_idx[bad]] = False
            if not bad.any():
                break
        except Exception:
            break
    return mask

# =============================================================================
# STEP 1 — PER-CLUSTER LOESS FIT
# =============================================================================

def fit_cluster(file_path):
    """
    Load a cluster CSV, apply quality filters and sigma-clipping, fit LOESS,
    approximate the LOESS curve with a cubic polynomial, and save a diagnostic
    plot.

    Returns a dict of fit parameters, or None if processing fails.
    """
    filename = os.path.basename(file_path)
    name     = filename.replace(".csv", "")
    print(f"\n>>> {name}")

    matched, prot_min = get_prot_limit(filename)
    if matched and prot_min is None:
        print("    Skipped (no Prot limit configured — set in CLUSTER_SETTINGS).")
        return None

    # Load
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as exc:
        print(f"    ERROR reading file: {exc}")
        return None
    df.columns = df.columns.str.strip()

    # Identify columns
    col_corr = find_col(df, ["bp_rp_corrected", "BP_RP_corrected"])
    col_raw  = find_col(df, ["bp_rp", "bprp", "bp-rp", "color", "BP-RP"])
    col_prot = find_col(df, ["Prot", "prot", "Period", "period", "P_rot"])
    col_logg = find_col(df, ["ISO_logg", "log_g", "logg"])

    if not all([col_corr, col_raw, col_prot]):
        print("    ERROR: Required columns missing (bp_rp_corrected, bp_rp, Prot).")
        return None

    for c in [col_corr, col_raw, col_prot, col_logg]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quality filters
    mask = (
        df[col_corr].between(-0.5, 4.0) &
        df[col_raw].notna() &
        df[col_prot].between(0.1, 60)
    )
    if col_logg:
        mask &= df[col_logg].between(LOG_G_MIN, LOG_G_MAX)
    if prot_min is not None:
        mask &= df[col_prot] > prot_min

    sub = df.loc[mask].copy()
    if len(sub) < 10:
        print(f"    Skipped: only {len(sub)} stars after filtering.")
        return None

    x     = sub[col_prot].values
    y_c   = sub[col_corr].values
    y_r   = sub[col_raw].values

    # Sigma-clipping
    keep  = iterative_sigma_clip(x, y_c, sigma=SIGMA_THRESHOLD, frac=FRAC_VALUE)
    x_c, y_cc, y_rc = x[keep], y_c[keep], y_r[keep]

    if len(x_c) < 5:
        print("    Skipped: too few stars after sigma-clipping.")
        return None

    idx      = np.argsort(x_c)
    xs, ycs  = x_c[idx], y_cc[idx]
    yrs      = y_rc[idx]

    # LOESS fits
    try:
        lo_c = sm.nonparametric.lowess(ycs, xs, frac=FRAC_VALUE, it=3)
        lo_r = sm.nonparametric.lowess(yrs, xs, frac=FRAC_VALUE, it=3)
        x_fit    = lo_c[:, 0]
        y_fit_c  = lo_c[:, 1]
        y_fit_r  = lo_r[:, 1]
    except Exception as exc:
        print(f"    ERROR in LOESS fit: {exc}")
        return None

    # Cubic polynomial approximation of the LOESS curve (Table 3)
    # Model: Color = A*Prot^3 + B*Prot^2 + C*Prot + D
    coeffs = np.polyfit(x_fit, y_fit_c, 3)
    A, B, C, D = coeffs

    print(f"    Stars used: {len(x_c)}")
    print(f"    Prot range: [{x_fit.min():.2f}, {x_fit.max():.2f}] days")
    print(f"    Poly fit  : Color = {A:.4e}*P^3 + {B:.4e}*P^2 + {C:.4f}*P + {D:.4f}")

    # Diagnostic plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x_c, y_rc, color="tomato",  s=25, alpha=0.25, label="Raw data")
    ax.scatter(x_c, y_cc, color="steelblue", s=30, alpha=0.7,
               edgecolors="k", lw=0.4, label="Corrected data")
    ax.plot(x_fit, y_fit_r, color="orange", ls="--", lw=1.8, label="LOESS (raw)")
    ax.plot(x_fit, y_fit_c, color="navy",   lw=2.5,          label="LOESS (corrected)")
    if prot_min:
        ax.axvline(prot_min, color="gold", ls=":", label=f"Prot min = {prot_min} d")
    ax.set_xlabel(r"$P_{\rm rot}$ (days)",            fontsize=12)
    ax.set_ylabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ (mag)", fontsize=12)
    ax.set_title(name, fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_loess_fit.png"), dpi=150)
    plt.close()

    return {
        "Cluster":           name,
        "Coeff_A (Prot^3)":  A,
        "Coeff_B (Prot^2)":  B,
        "Coeff_C (Prot^1)":  C,
        "Coeff_D (intercept)": D,
        "Prot_min":          x_fit.min(),
        "Prot_max":          x_fit.max(),
        "N_stars":           len(x_c),
        "Equation":          "Color = A*Prot^3 + B*Prot^2 + C*Prot + D",
    }


print("=" * 60)
print("STEP 1 — Per-cluster LOESS fit")
print("=" * 60)

csv_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.csv"))
             if "gyro_fit" not in f and "isochrone_match" not in f]

all_params = []
for f in csv_files:
    res = fit_cluster(f)
    if res:
        all_params.append(res)

if not all_params:
    raise SystemExit("No clusters fitted. Check INPUT_DIR and CLUSTER_SETTINGS.")

params_df = pd.DataFrame(all_params)
param_csv = os.path.join(OUTPUT_DIR, "gyro_fit_parameters.csv")
params_df.to_csv(param_csv, index=False)
print(f"\nFit parameters saved: {param_csv}")
print(params_df.to_string(index=False))

# =============================================================================
# STEP 2 — COMPOSITE GYROCHRONOLOGY PLOT (Figure 7)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — Composite gyrochronology diagram")
print("=" * 60)

ages_db  = [v["age"] for v in CLUSTER_DB.values()]
norm     = mcolors.LogNorm(vmin=min(ages_db), vmax=max(ages_db))
cmap     = plt.cm.turbo

fig, ax = plt.subplots(figsize=(11, 8))

for _, row in params_df.sort_values("Prot_min").iterrows():
    # Match to CLUSTER_DB
    key = row["Cluster"].lower().replace("_", "").replace(" ", "")
    db  = next((v for k, v in CLUSTER_DB.items()
                if k.replace("_", "") in key), None)
    if db is None:
        print(f"  WARNING: '{row['Cluster']}' not found in CLUSTER_DB — skipped.")
        continue

    A, B, C, D = (row["Coeff_A (Prot^3)"], row["Coeff_B (Prot^2)"],
                  row["Coeff_C (Prot^1)"], row["Coeff_D (intercept)"])

    prot_vals  = np.linspace(row["Prot_min"], row["Prot_max"], 300)
    color_vals = A * prot_vals**3 + B * prot_vals**2 + C * prot_vals + D

    feh_str = f"+{db['feh']:.3f}" if db["feh"] >= 0 else f"{db['feh']:.3f}"
    label   = f"{db['label']}  [Fe/H] = {feh_str}"

    ax.plot(color_vals, prot_vals,
            color=cmap(norm(db["age"])), lw=2.5, alpha=0.9, label=label)
    print(f"  Plotted: {db['label']}  (age = {db['age']} Myr)")

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Age (Myr)", fontsize=12)

ax.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ (mag)", fontsize=13)
ax.set_ylabel(r"$P_{\rm rot}$ (days)",               fontsize=13)
ax.legend(loc="upper left", fontsize=9, frameon=True,
          facecolor="white", edgecolor="gray", framealpha=0.9)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "gyrochronology_composite.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nFigure saved: {fig_path}")
