"""
04_praesepe_validation.py
-------------------------
Statistical validation of the Veq derivation pipeline using the Praesepe
cluster (M44), which contains stars with both spectroscopic v sin i and
photometric rotation period (Prot) measurements.

Produces:
  - Descriptive statistics for both samples
  - Two-sample KS test and global Mann-Whitney U test
  - Bin-by-bin median comparison with per-bin Mann-Whitney U tests
  - median(v sin i) / median(Veq) ratio vs. the isotropic expectation (pi/4)
  - Figure 8 from Usta et al. (2025): scatter + running medians, bin
    comparison bar chart, and empirical CDF panel
  - LaTeX table (Table 4) printed to stdout

Reference: Usta et al. (2025), PASP - Section 3.2 and Table 4
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================

# Paths to the two Gold Group (v sin i) catalogs for Praesepe.
# Place your CSV files in the data/ folder and update the filenames below.
OBS_FILES = [
    os.path.join("data", "PRAESEPE_vsini_1.csv"),
    os.path.join("data", "PRAESEPE_vsini_2.csv"),
]

# Path to the Silver Group (Veq) output from 02_veq_calculation.py
THEO_FILE = os.path.join("data", "output", "PRAESEPE_isochrone_match.csv")

# Praesepe extinction parameters
AV     = 0.08
AG     = 0.836 * AV
E_BPRP = AG / 1.862          # E(BP-RP) reddening correction

# Output figure path
OUTPUT_FIG = os.path.join("data", "output", "praesepe_validation.png")

print(f"Praesepe: Av = {AV},  E(BP-RP) = {E_BPRP:.4f}\n")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_col(df, candidates):
    """Return the first column in `df` whose name matches any candidate
    (case-insensitive). Returns None if no match is found."""
    for cand in candidates:
        match = next((c for c in df.columns if c.lower() == cand.lower()), None)
        if match:
            return match
    return None


def load_obs(file_list, e_bprp):
    """
    Load and concatenate Gold Group (v sin i) catalogs.

    Expects columns: BP-RP color and v sin i (accepted names listed below).
    Applies reddening correction to produce bp_rp_corrected.

    Returns pd.DataFrame with columns [color_raw, bp_rp_corrected, vsini].
    """
    frames = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"  WARNING: File not found -> {f}")
            continue
        try:
            try:
                df = pd.read_csv(f)
            except Exception:
                df = pd.read_csv(f, sep=";")
            df.columns = df.columns.str.strip()

            col_color = find_col(df, ["bp_rp", "bprp", "bp-rp", "color"])
            col_vsini = find_col(df, ["vsini", "v_sini", "rotational_velocity"])

            if col_color and col_vsini:
                df[col_color]  = pd.to_numeric(df[col_color],  errors="coerce")
                df[col_vsini]  = pd.to_numeric(df[col_vsini],  errors="coerce")
                df["bp_rp_corrected"] = df[col_color] - e_bprp
                sub = df[[col_color, "bp_rp_corrected", col_vsini]].dropna()
                frames.append(sub)
                print(f"  + {os.path.basename(f)}: {len(sub)} stars loaded")
            else:
                print(f"  - {os.path.basename(f)}: required columns not found, skipped")

        except Exception as exc:
            print(f"  ERROR ({os.path.basename(f)}): {exc}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_theo(filepath, e_bprp):
    """
    Load Silver Group (Veq) catalog produced by 02_veq_calculation.py.

    Applies reddening correction if a pre-corrected color column is absent.

    Returns pd.DataFrame with columns [bp_rp_final, Calc_Veq].
    """
    if not os.path.exists(filepath):
        print(f"  ERROR: Theoretical file not found -> {filepath}")
        return None
    try:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, sep=";")
        df.columns = df.columns.str.strip()

        col_corr  = find_col(df, ["bp_rp_corrected"])
        col_raw   = find_col(df, ["bp_rp", "bprp", "bp-rp"])
        col_veq   = find_col(df, ["Calc_Veq", "Calc_V_eq", "V_eq", "veq"])

        if col_veq is None:
            print("  ERROR: Veq column not found in theoretical file.")
            return None

        df[col_veq] = pd.to_numeric(df[col_veq], errors="coerce")

        if col_corr:
            df["bp_rp_final"] = pd.to_numeric(df[col_corr], errors="coerce")
        elif col_raw:
            df[col_raw] = pd.to_numeric(df[col_raw], errors="coerce")
            df["bp_rp_final"] = df[col_raw] - e_bprp
        else:
            print("  ERROR: Color column not found in theoretical file.")
            return None

        result = df[["bp_rp_final", col_veq]].dropna()
        print(f"  + Theoretical file: {len(result)} stars loaded")
        return result

    except Exception as exc:
        print(f"  ERROR loading theoretical file: {exc}")
        return None

# =============================================================================
# LOAD DATA
# =============================================================================

print("--- Loading data ---")
df_obs  = load_obs(OBS_FILES, E_BPRP)
df_theo = load_theo(THEO_FILE, E_BPRP)

if df_obs.empty or df_theo is None or df_theo.empty:
    raise SystemExit("Cannot continue: one or both datasets are empty.")

vsini_vals = df_obs.iloc[:, -1].values.astype(float)
veq_vals   = df_theo.iloc[:, -1].values.astype(float)
color_obs  = df_obs["bp_rp_corrected"].values.astype(float)
color_theo = df_theo["bp_rp_final"].values.astype(float)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS SUMMARY")
print("=" * 60)

# --- Descriptive statistics ---
print("\n[1] Descriptive Statistics")
header = f"{'Statistic':<20} {'v sin i (obs)':<22} {'Veq (theo)':<20}"
print(header)
print("-" * 60)
for label, fn in [
    ("N",        lambda x: str(len(x))),
    ("Mean",     lambda x: f"{np.mean(x):.2f} km/s"),
    ("Median",   lambda x: f"{np.median(x):.2f} km/s"),
    ("Std Dev",  lambda x: f"{np.std(x):.2f} km/s"),
    ("Min",      lambda x: f"{np.min(x):.2f} km/s"),
    ("Max",      lambda x: f"{np.max(x):.2f} km/s"),
]:
    print(f"{label:<20} {fn(vsini_vals):<22} {fn(veq_vals):<20}")

# --- KS test ---
print("\n[2] Two-Sample Kolmogorov-Smirnov Test")
ks_stat, ks_p = stats.ks_2samp(vsini_vals, veq_vals)
print(f"  KS statistic : {ks_stat:.4f}")
print(f"  p-value      : {ks_p:.4e}")
if ks_p < 0.05:
    print("  --> Distributions are significantly different (expected: vsini < Veq due to projection).")
else:
    print("  --> No significant difference detected.")

# --- Mann-Whitney U test (global) ---
print("\n[3] Mann-Whitney U Test (global, two-sided)")
mw_stat, mw_p = stats.mannwhitneyu(vsini_vals, veq_vals, alternative="two-sided")
print(f"  U statistic  : {mw_stat:.2f}")
print(f"  p-value      : {mw_p:.4e}")
if mw_p < 0.05:
    print("  --> Medians are significantly different.")
else:
    print("  --> No significant difference in medians.")

# --- sin(i) proxy ---
print("\n[4] median(v sin i) / median(Veq) ratio")
ratio = np.median(vsini_vals) / np.median(veq_vals)
deviation = abs(ratio - np.pi / 4) / (np.pi / 4) * 100
print(f"  Observed ratio          : {ratio:.3f}")
print(f"  Isotropic expectation   : {np.pi/4:.3f}  (= pi/4)")
print(f"  Deviation from pi/4     : {deviation:.1f}%")

# --- Bin-by-bin analysis ---
print("\n[5] Bin-by-Bin Median Comparison")

color_all = np.concatenate([color_obs, color_theo])
bin_edges = np.linspace(np.percentile(color_all, 5),
                        np.percentile(color_all, 95), 7)

bin_results = []
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask_o = (color_obs  >= lo) & (color_obs  < hi)
    mask_t = (color_theo >= lo) & (color_theo < hi)
    if mask_o.sum() < 3 or mask_t.sum() < 3:
        continue
    med_o = np.median(vsini_vals[mask_o])
    med_t = np.median(veq_vals[mask_t])
    _, p_bin = stats.mannwhitneyu(vsini_vals[mask_o], veq_vals[mask_t],
                                  alternative="two-sided")
    bin_results.append({
        "bin":       f"[{lo:.2f}, {hi:.2f}]",
        "n_obs":     mask_o.sum(),
        "n_theo":    mask_t.sum(),
        "med_vsini": med_o,
        "med_veq":   med_t,
        "ratio":     med_o / med_t if med_t > 0 else np.nan,
        "p_value":   p_bin,
    })

df_bins = pd.DataFrame(bin_results)

print(f"\n  {'Color Bin':<18} {'N_obs':>6} {'N_theo':>7} "
      f"{'Med vsini':>10} {'Med Veq':>9} {'Ratio':>7} {'p-value':>10}")
print("  " + "-" * 72)
for _, row in df_bins.iterrows():
    sig = "*" if row["p_value"] < 0.05 else " "
    print(f"  {row['bin']:<18} {row['n_obs']:>6} {row['n_theo']:>7} "
          f"{row['med_vsini']:>9.2f}  {row['med_veq']:>8.2f}  "
          f"{row['ratio']:>6.3f}  {row['p_value']:>9.4e} {sig}")
print("  (* = significant at p < 0.05)")

if len(df_bins) >= 3:
    sp_r, sp_p = stats.spearmanr(df_bins["med_vsini"], df_bins["med_veq"])
    print(f"\n  Spearman r (bin medians): {sp_r:.3f}  p = {sp_p:.4f}")

# =============================================================================
# FIGURE (Figure 8 in Usta et al. 2025)
# =============================================================================

def running_median(ax, x, y, color, ls, label, nbins=8):
    """Overlay a running-median line on an existing axis."""
    edges = np.linspace(np.percentile(x, 2), np.percentile(x, 98), nbins + 1)
    cx, my = [], []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() >= 3:
            cx.append((edges[i] + edges[i + 1]) / 2)
            my.append(np.median(y[mask]))
    if cx:
        ax.plot(cx, my, color=color, ls=ls, lw=2.2, marker="o",
                markersize=5, label=label, zorder=5)


plt.style.use("default")
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, :])   # Panel a: scatter + running medians
ax2 = fig.add_subplot(gs[1, 0])   # Panel b: bin median bars
ax3 = fig.add_subplot(gs[1, 1])   # Panel c: empirical CDF

fig.patch.set_facecolor("white")

# Panel a
ax1.scatter(color_obs, vsini_vals, color="dodgerblue", s=40, alpha=0.55,
            edgecolors="black", linewidth=0.4,
            label=r"Observation ($v\sin i$)", zorder=2)
ax1.scatter(color_theo, veq_vals, color="crimson", marker="x", s=30,
            alpha=0.65, label=r"Theoretical ($V_{\rm eq}$)", zorder=1)

running_median(ax1, color_obs,  vsini_vals, "navy",    "-",
               r"Running median ($v\sin i$)")
running_median(ax1, color_theo, veq_vals,   "darkred", "--",
               r"Running median ($V_{\rm eq}$)")

ax1.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ (mag)", fontsize=13)
ax1.set_ylabel(r"Velocity (km s$^{-1}$)",            fontsize=13)
ax1.set_title(r"a) Praesepe: Observed $v\sin i$ vs. Theoretical $V_{\rm eq}$",
              fontsize=13, fontweight="bold", loc="left")
ax1.legend(fontsize=9.5, frameon=True, facecolor="white",
           edgecolor="black", ncol=2)
ax1.set_ylim(0, max(vsini_vals.max(), veq_vals.max()) * 1.12)

# Panel b
if not df_bins.empty:
    x_pos = np.arange(len(df_bins))
    w = 0.35
    ax2.bar(x_pos - w / 2, df_bins["med_vsini"], w,
            color="dodgerblue", edgecolor="black", lw=0.7,
            label=r"Median $v\sin i$")
    ax2.bar(x_pos + w / 2, df_bins["med_veq"], w,
            color="crimson", edgecolor="black", lw=0.7,
            label=r"Median $V_{\rm eq}$")
    for j, (_, row) in enumerate(df_bins.iterrows()):
        if row["p_value"] < 0.05:
            ymax = max(row["med_vsini"], row["med_veq"])
            ax2.text(j, ymax + 0.5, "*", ha="center", va="bottom",
                     fontsize=12, color="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [r.replace("[", "").replace("]", "") for r in df_bins["bin"]],
        rotation=35, ha="right", fontsize=7.5)
    ax2.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ bin (mag)", fontsize=11)
    ax2.set_ylabel(r"Median velocity (km s$^{-1}$)",          fontsize=11)
    ax2.set_title("b) Bin-by-Bin Median Comparison\n(* = p < 0.05)",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="black")

# Panel c
def plot_ecdf(ax, data, color, ls, label):
    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, color=color, lw=2, linestyle=ls, label=label, where="post")

plot_ecdf(ax3, vsini_vals, "dodgerblue", "-",  r"$v\sin i$ (obs)")
plot_ecdf(ax3, veq_vals,   "crimson",    "--", r"$V_{\rm eq}$ (theo)")
ax3.set_xlabel(r"Velocity (km s$^{-1}$)",  fontsize=11)
ax3.set_ylabel("Cumulative fraction",      fontsize=11)
ax3.set_title(
    f"c) Empirical CDF\nKS stat = {ks_stat:.3f},  p = {ks_p:.2e}",
    fontsize=11, fontweight="bold", loc="left")
ax3.legend(fontsize=9.5, frameon=True, facecolor="white", edgecolor="black")
ax3.set_ylim(0, 1.05)

os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
print(f"\nFigure saved: {OUTPUT_FIG}")

# =============================================================================
# LaTeX TABLE (Table 4 in Usta et al. 2025)
# =============================================================================

print("\n" + "=" * 60)
print("LaTeX TABLE (Table 4 — ready for Overleaf)")
print("=" * 60)

latex = r"""
\begin{table}[ht]
\centering
\caption{Bin-by-bin comparison of median observed $v\sin i$ and theoretical
         $V_{\rm eq}$ for Praesepe. The last column gives the Mann--Whitney
         $p$-value for each colour bin; asterisks indicate $p < 0.05$.}
\label{tab:vsini_veq_bins}
\begin{tabular}{lrrrrr}
\hline\hline
$(G_{\rm BP}{-}G_{\rm RP})_0$ & $N_{\rm obs}$ & $N_{\rm theo}$ &
Median $v\sin i$ & Median $V_{\rm eq}$ & $p$-value \\
(mag) & & & (km\,s$^{-1}$) & (km\,s$^{-1}$) & \\
\hline
"""
for _, row in df_bins.iterrows():
    sig = r"$^*$" if row["p_value"] < 0.05 else ""
    latex += (f"{row['bin']} & {row['n_obs']} & {row['n_theo']} & "
              f"{row['med_vsini']:.1f} & {row['med_veq']:.1f} & "
              f"{row['p_value']:.2e}{sig} \\\\\n")

latex += r"""\hline
\multicolumn{6}{l}{$^*$ Significant at $p < 0.05$ (Mann--Whitney U test).} \\
\end{tabular}
\end{table}
"""

latex += (
    f"\n% Global statistics (add to main text):\n"
    f"% KS test: D = {ks_stat:.4f}, p = {ks_p:.2e}\n"
    f"% Mann-Whitney U (global): U = {mw_stat:.0f}, p = {mw_p:.2e}\n"
    f"% median(vsini) / median(Veq) = {ratio:.3f}  "
    f"[isotropic expectation: pi/4 = {np.pi/4:.3f}]\n"
)
print(latex)
