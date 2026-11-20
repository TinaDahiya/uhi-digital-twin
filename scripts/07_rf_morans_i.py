
"""Spatial autocorrelation analysis of RF residuals."""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from libpysal.weights import KNN
from esda.moran import Moran


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

RESIDUAL_PATH = (
    ROOT / "models" / "rf_analysis" / "rf_residuals.csv"
)

OUTPUT_DIR = ROOT / "models" / "rf_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD RESIDUALS
# ==================================================

print("Loading residuals...")

df = pd.read_csv(RESIDUAL_PATH)

df = df.dropna(subset=["lon", "lat", "residual"])
df = df.reset_index(drop=True)

print(f"Observations: {len(df)}")


# ==================================================
# SPATIAL WEIGHTS
# ==================================================

print("\nCreating spatial neighbour weights...")

coords = df[["lon", "lat"]].to_numpy()

# Connect every cell to its 8 nearest test cells
weights = KNN.from_array(coords, k=8)

# Row-standardized weights
weights.transform = "R"

print("Spatial weights created.")


# ==================================================
# MORAN'S I
# ==================================================

print("\nCalculating Moran's I...")

residuals = df["residual"].to_numpy()

moran = Moran(
    residuals,
    weights,
    permutations=999,
)

results = {
    "observations": int(len(df)),
    "neighbours": 8,
    "moran_I": float(moran.I),
    "expected_I": float(moran.EI),
    "permutation_p_value": float(moran.p_sim),
    "z_score": float(moran.z_sim),
}

# Interpretation
if moran.p_sim < 0.05 and moran.I > moran.EI:
    interpretation = "Significant positive spatial autocorrelation."

elif moran.p_sim < 0.05 and moran.I < moran.EI:
    interpretation = "Significant negative spatial autocorrelation."

else:
    interpretation = "No significant spatial autocorrelation detected."

results["interpretation"] = interpretation

print("\nMoran's I results:")

for key, value in results.items():
    print(f"{key}: {value}")


# ==================================================
# SAVE RESULTS
# ==================================================

with open(
    OUTPUT_DIR / "morans_i_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(results, f, indent=4)


# ==================================================
# MORAN SCATTERPLOT
# ==================================================

# Spatial lag = weighted average of neighbouring residuals
from libpysal.weights import lag_spatial

spatial_lag = lag_spatial(weights, residuals)

plt.figure(figsize=(8, 6))

plt.scatter(
    residuals,
    spatial_lag,
    alpha=0.5,
)

plt.axhline(0, linestyle="--", color="black")
plt.axvline(0, linestyle="--", color="black")

plt.xlabel("Standardized Residual")
plt.ylabel("Spatial Lag of Residual")
plt.title(f"Moran Scatterplot (I = {moran.I:.3f})")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "morans_scatterplot.png",
    dpi=300,
)

plt.close()

print("\nMoran's I analysis completed.")
print(f"Results saved in: {OUTPUT_DIR}")
