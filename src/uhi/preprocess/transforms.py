"""Step G: log1p on skewed features; train-only winsorization of features."""
import numpy as np
import pandas as pd


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["NTL_log"] = np.log1p(d["NTL"].clip(lower=0))
    d["Pop_log"] = np.log1p(d["Pop"].clip(lower=0))
    return d


class Winsorizer:
    """Bounds are fit on the TRAIN split only and applied to both splits (spec section 7 / 8.3)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper, self.bounds_ = lower, upper, {}

    def fit(self, train: pd.DataFrame, cols):
        self.bounds_ = {c: (float(train[c].quantile(self.lower)), float(train[c].quantile(self.upper))) for c in cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c, (lo, hi) in self.bounds_.items():
            d[c] = d[c].clip(lo, hi)
        return d


def year_center(df: pd.DataFrame, cols, train_mask):
    """Subtract each year's TRAIN-cell mean (test cells never contribute). Raw values are kept as `<col>__raw` in the
    full table only. Returns (frame, {year: {col: mean}})."""
    d = df.copy()
    means = df.loc[train_mask].groupby("year")[list(cols)].mean()
    missing = set(df.year.unique()) - set(means.index)
    if missing:
        raise ValueError(f"No training cells in year(s) {sorted(missing)}; cannot centre.")
    for c in cols:
        d[c + "__raw"] = d[c]
        d[c] = d[c] - d["year"].map(means[c])
    return d, {int(y): {c: float(means.loc[y, c]) for c in cols} for y in means.index}
