# Stellar Rotation in Open Clusters

Analysis pipeline for:

> **Usta A., Kayhan C., Akkaya Oralhan I.**
> *Stellar Rotation in Open Clusters: Exploring Rotational Properties in 21 Clusters with Different Ages and Metallicities*
> Submitted to PASP (2025)

---

## Overview

This repository contains the Python scripts used to derive equatorial rotational
velocities (V_eq) from photometric rotation periods (P_rot) for the **Silver Group**
clusters in the study above.

The pipeline consists of four scripts:

| Script | Section in paper | Description |
|---|---|---|
| `01_isochrone_matching.py` | §2.3, Eq. 1–2 | Matches Gaia DR3 photometry to PARSEC isochrones in CMD space; derives T_eff, L, and R |
| `02_veq_calculation.py` | §2.3, Eq. 3 | Computes equatorial velocities: V_eq = 2πR / P_rot |
| `04_praesepe_validation.py` | §3.2, Table 4, Fig. 8 | Statistical validation via KS test, Mann-Whitney U, bin-by-bin median comparison, and empirical CDF; outputs LaTeX table |

---

## Requirements

```
Python >= 3.8
pandas
numpy
```

Install with:

```bash
pip install pandas numpy
```

---

## Data

The scripts expect the following files inside a `data/` folder:

```
data/
├── CLUSTERNAME.csv          # Gaia DR3 member catalog (must include G, bp_rp or BP_RP, and Prot)
└── CLUSTERNAME_ISO.txt      # PARSEC isochrone file (space-separated, downloaded from CMD web interface)
```

**Observation CSV columns required:**

| Column | Description |
|---|---|
| `G` | Gaia G-band apparent magnitude |
| `bp_rp` or `BP_RP` | Gaia BP−RP color index |
| `Prot` (or `Per`, `Period`, `P`) | Photometric rotation period (days) |

**PARSEC isochrone columns required:** `Gmag` (or `G_fSBmag`), `G_BPmag`, `G_RPmag`, `logTe`, `logL`

Isochrone files can be obtained from the PARSEC CMD web interface:
http://stev.oapd.inaf.it/cgi-bin/cmd

> **Note:** The raw data files used in the paper are not included in this repository
> because they originate from third-party catalogs. Please refer to Table 1 of the
> paper for the original data sources.

---

## Usage

```bash
# Step 1: match stars to isochrones and derive Teff, L, R
python 01_isochrone_matching.py

# Step 2: compute equatorial velocities
python 02_veq_calculation.py

# Step 3: statistical validation using Praesepe
python 04_praesepe_validation.py
```

Output CSVs will be written to `data/output/`.

---

## Configuration

At the top of each script you can adjust:

- `THRESHOLD_VAL` — maximum CMD distance (in mag) for an isochrone match (default: 0.15)
- `BASE_DIR` — path to your data folder
- `clusters` — list of cluster parameters (name, distance modulus, A_V, filenames)

---

## Citation

If you use this pipeline, please cite:

```
Usta A., Kayhan C., Akkaya Oralhan I. (2025)
Stellar Rotation in Open Clusters: Exploring Rotational Properties
in 21 Clusters with Different Ages and Metallicities
PASP (submitted)
```

---

## License

MIT License — see `LICENSE` file.
