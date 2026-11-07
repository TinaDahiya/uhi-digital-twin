"""Spec section 12 gates (plus multi-year checks). Status: PASS / WARN / FAIL / SKIP / MANUAL."""
import pandas as pd

from .schema import DW_ALL, TARGET


def _g(name, status, detail):
    return {"gate": name, "status": status, "detail": detail}


def run_gates(c: dict) -> list[dict]:
    g, d, ml, cfg = [], c["clean_df"], c["ml"], c["gates_cfg"]
    removed = sum(s["removed"] for s in c["clean_log"]["steps"])
    g.append(_g("row_count", "PASS" if c["raw_rows"] - removed == len(ml) else "FAIL",
                f"raw={c['raw_rows']}, removed={removed}, final={len(ml)}"))
    n_null = int(ml[[TARGET] + c["feats"]].isna().sum().sum())
    g.append(_g("nulls", "PASS" if n_null == 0 else "FAIL", f"{n_null} nulls in target+features"))

    lo, hi = cfg["lst_range"]
    g.append(_g("lst_sanity", "PASS" if d.LST.min() >= lo and d.LST.max() <= hi else "FAIL",
                f"cell-mean LST {d.LST.min():.1f}..{d.LST.max():.1f} C (allowed {lo}..{hi}); per-year medians "
                + ", ".join(f"{y}: {v:.1f}" for y, v in d.groupby('year').LST.median().items())))
    m0, m1 = cfg["suhii_median_range"]
    meds = d.groupby("year")[TARGET].median()
    g.append(_g("suhii_median", "PASS" if meds.between(m0, m1).all() else "WARN",
                f"per-year medians " + ", ".join(f"{y}: {v:.2f}" for y, v in meds.items()) +
                f" (spec band {m0}..{m1} comes from Mumbai - calibrate for this city)"))
    mx = d[TARGET].abs().max()
    g.append(_g("suhii_extremes", "PASS" if mx <= cfg["suhii_abs_max"] else "FAIL", f"max |SUHII| {mx:.2f} (limit {cfg['suhii_abs_max']})"))

    pm, exp = float(d.Pop.max()), c.get("expected_pop_max")
    g.append(_g("population_max", "SKIP" if exp is None else ("PASS" if pm <= exp * 1.001 else "FAIL"),
                f"cell-mean Pop max={pm:.0f}" + (" - set preprocess.expected_pop_max from the GPW source" if exp is None else f" vs source max {exp}")))
    nmin, nmax = d.NTL_log.min(), d.NTL_log.max()
    g.append(_g("ntl", "FAIL" if nmin < 0 else ("PASS" if nmax <= cfg["ntl_log_max"] else "WARN"), f"NTL_log {nmin:.2f}..{nmax:.2f}"))

    frac = float(((d[DW_ALL].sum(axis=1) - 1).abs() <= 0.05).mean())
    g.append(_g("dynamic_world_sum", "PASS" if frac >= 0.95 else "WARN", f"{frac:.1%} of cell-years have class probabilities summing to 1 +/- 0.05"))
    big = {k: v for k, v in c["dw_dropped_shares"].items() if v > c["dw_max_share"]}
    if not c["dw_dropped_shares"]:
        g.append(_g("dw_dropped_class_share", "PASS", "no Dynamic World class dropped up front (only the compositional reference)"))
    else:
        g.append(_g("dw_dropped_class_share", "PASS" if not big else "WARN",
                    "dropped-class mean shares: " + ", ".join(f"{k}={v:.4f}" for k, v in c["dw_dropped_shares"].items()) +
                    (f" -> NOT negligible: {big}" if big else f" (all <= {c['dw_max_share']})")))

    if c.get("expected_years"):
        have = sorted(int(y) for y in d.year.unique())
        g.append(_g("years_present", "PASS" if have == sorted(c["expected_years"]) else "FAIL", f"expected {sorted(c['expected_years'])}, have {have}"))
    per = d.groupby("year").size()
    ratio = per.min() / per.max()
    g.append(_g("cells_per_year", "PASS" if ratio >= cfg["min_cell_ratio_across_years"] else "WARN",
                "cells per year " + ", ".join(f"{y}: {n}" for y, n in per.items()) + f" (min/max={ratio:.2f})"))

    tr, te = set(ml.loc[ml.split == "train", "cell_id"]), set(ml.loc[ml.split == "test", "cell_id"])
    g.append(_g("split_cell_integrity", "PASS" if not tr & te else "FAIL",
                f"{len(tr)} train cells, {len(te)} test cells, overlap={len(tr & te)} (a cell must never sit in both splits in any year)"))
    bc = ml.groupby(["split", "block_id"]).size()
    small = bc[bc < c["min_rows_per_block"]]
    g.append(_g("spatial_coverage", "PASS" if small.empty else "WARN",
                f"blocks: train={int((bc.index.get_level_values(0) == 'train').sum())}, test={int((bc.index.get_level_values(0) == 'test').sum())}; "
                f"blocks with < {c['min_rows_per_block']} rows: {len(small)}; train row share={c['split_info']['train_row_frac']:.3f}"))
    il = c.get("impute_log")
    if il is None:
        g.append(_g("imputation", "SKIP", "imputation disabled: rows with a missing feature were removed"))
    else:
        worst = max(((u, v["share_of_rows"]) for u, v in il["units"].items()), key=lambda t: t[1], default=("-", 0.0))
        total = sum(v["imputed"] for v in il["units"].values())
        g.append(_g("imputation", "PASS" if worst[1] <= c["impute_warn"] else "WARN",
                    f"{il['rows_with_imputed_values']} rows had values imputed ({total} values, nearest cell, same year+split, "
                    f"<= {il['max_distance_km']} km); most-imputed feature: {worst[0]} ({worst[1]:.1%} of rows; warn above {c['impute_warn']:.0%}); "
                    f"removed instead of imputed: {il['rows_removed_too_many_missing']} (too many missing), "
                    f"{il['rows_removed_no_donor_in_range']} (no donor in range). Target never imputed."))
    prov = c.get("provenance")
    if not prov:
        g.append(_g("period_coverage", "SKIP", "no export log found; cannot verify the months each year covers"))
    else:
        months = {int(y): e.get("months_in_window") for y, e in prov.items() if e.get("months_in_window")}
        ok = len(set(months.values())) == 1
        g.append(_g("period_coverage", "PASS" if ok else "WARN",
                    "months per composite: " + ", ".join(f"{y}: {m}" for y, m in sorted(months.items())) +
                    ("" if ok else " -> years cover DIFFERENT calendar months, so between-year differences mix season with year; "
                                   "set align_months: true for like-for-like composites")))
    g.append(_g("reproducibility", "PASS" if c["repro_ok"] else "FAIL", "two independent runs give an identical table" if c["repro_ok"] else "runs differ"))
    g.append(_g("literature_check", "MANUAL", "Compare the LST distribution with published pre-monsoon values for Delhi."))
    return g
