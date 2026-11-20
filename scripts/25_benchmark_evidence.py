"""Summarize a completed script-24 run; paired spatial-block bootstrap on fixed OOF predictions."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(run, draws=2000):
    meta = json.loads((run / "metadata.json").read_text())
    if meta["smoke"]:
        raise ValueError("Smoke results cannot be used as benchmark evidence.")
    oof = pd.read_csv(run / "oof_predictions.csv")
    folds = pd.read_csv(run / "fold_results.csv")
    expected = len(oof[["cell_id", "year", "block_id"]].drop_duplicates())
    sizes = oof.groupby(["model", "variant"]).size()
    if len(sizes) != 16 or not sizes.eq(expected).all() or len(folds) != 80:
        raise ValueError("Expected complete 4-model x 4-variant x 5-fold run.")
    keys = ["cell_id", "year", "block_id", "fold"]
    if oof.duplicated(keys + ["model", "variant"]).any():
        raise ValueError("Duplicate OOF keys.")
    blocks = sorted(oof.block_id.unique())
    rng = np.random.default_rng(20261117)
    counts = rng.multinomial(len(blocks), np.ones(len(blocks)) / len(blocks), size=draws)
    records = []
    for family in sorted(oof.model.unique()):
        proposed = oof[oof.model.eq(family) & oof.variant.eq("cell_context_contrast")]
        for variant in ["cell_only", "cell_context", "cell_contrast"]:
            base = oof[oof.model.eq(family) & oof.variant.eq(variant)]
            pair = proposed.merge(base, on=keys, suffixes=("_p", "_b"), validate="one_to_one")
            if len(pair) != expected or not np.array_equal(pair.SUHII_p, pair.SUHII_b):
                raise ValueError("Mismatched validation targets/identities.")
            pair["sse_p"] = (pair.SUHII_p - pair.prediction_p) ** 2
            pair["sse_b"] = (pair.SUHII_b - pair.prediction_b) ** 2
            group = pair.groupby("block_id").agg(
                n=("sse_p","size"), p=("sse_p","sum"), b=("sse_b","sum")).reindex(blocks)
            denom = counts @ group.n.to_numpy()
            sampled = np.sqrt((counts @ group.p.to_numpy()) / denom) - np.sqrt(
                (counts @ group.b.to_numpy()) / denom)
            point = float(np.sqrt(pair.sse_p.mean()) - np.sqrt(pair.sse_b.mean()))
            lo, hi = np.quantile(sampled, [.025, .975])
            records.append(dict(model=family, baseline=variant,
                proposed="cell_context_contrast", pooled_rmse_delta=point,
                block_bootstrap_delta_lo=float(lo), block_bootstrap_delta_hi=float(hi),
                draws=draws, blocks=len(blocks)))
    out = pd.DataFrame(records)
    out.to_csv(run / "block_bootstrap_comparisons.csv", index=False)
    (run / "bootstrap_scope.txt").write_text(
        "Paired resampling of geographic blocks, keeping all years of a cell together. "
        "Fixed OOF predictions only: models are not refitted, overlapping-fold fit uncertainty "
        "and dependence between nearby blocks are not captured. Intervals are exploratory "
        "and conditional; they do not establish statistical significance, independence, "
        "published-method superiority, or novelty. No multiplicity correction. "
        "Feature choice used these folds previously.", encoding="utf-8")
    print(out.round(6).to_string(index=False))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--draws", type=int, default=2000)
    args = p.parse_args()
    if args.draws < 100: p.error("--draws must be >=100")
    summarize(args.run, args.draws)
