"""Spatial block split. Blocks depend ONLY on location, so a cell is in the same split in every year (no temporal
leakage), and the assignment is stable when new years are appended (frozen in the state file)."""
import hashlib

import numpy as np
import pandas as pd


def compute_bounds(df):
    return [float(df.lon.min()), float(df.lon.max()), float(df.lat.min()), float(df.lat.max())]


def assign_blocks(df: pd.DataFrame, n_blocks: int, bounds) -> pd.Series:
    lo0, lo1, la0, la1 = bounds
    ex = np.linspace(lo0, lo1, n_blocks + 1)[1:-1]
    ey = np.linspace(la0, la1, n_blocks + 1)[1:-1]
    ix = np.clip(np.digitize(df.lon, ex), 0, n_blocks - 1)
    iy = np.clip(np.digitize(df.lat, ey), 0, n_blocks - 1)
    return pd.Series([f"b{a:02d}{b:02d}" for a, b in zip(ix, iy)], index=df.index)


def assign_split(df: pd.DataFrame, train_frac=0.7, seed=42, frozen=None):
    """Whole blocks -> train/test. Blocks already in `frozen` keep their side. New blocks are visited in a
    seed-hash order (independent of how many other blocks exist) and sent to train while the running train share of
    rows is below `train_frac`. Returns (split Series, full block map incl. frozen blocks no longer present)."""
    frozen = dict(frozen or {})
    counts = df.groupby("block_id").size()
    bmap = {b: frozen[b] for b in counts.index if b in frozen}
    assigned = int(sum(counts[b] for b in bmap))
    train_rows = int(sum(counts[b] for b, s in bmap.items() if s == "train"))
    new = sorted((b for b in counts.index if b not in bmap),
                 key=lambda b: hashlib.sha256(f"{seed}:{b}".encode()).hexdigest())
    for b in new:
        to_train = assigned == 0 or train_rows / assigned < train_frac
        bmap[b] = "train" if to_train else "test"
        assigned += int(counts[b])
        train_rows += int(counts[b]) if to_train else 0
    full = {**frozen, **bmap}
    return df["block_id"].map(bmap), dict(sorted(full.items()))
