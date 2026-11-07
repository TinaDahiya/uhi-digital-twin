"""Step H: candidate features, zero-variance, Spearman and VIF filters.
Every filter is fit on the TRAIN split only, and every removal carries a human-readable reason."""
import numpy as np
import pandas as pd

from .schema import BASE_FEATURES, DW_ALL, LEAKY


def candidate_features(dw_drop, dw_reference):
    """Returns (candidate feature list, [(column, kind)] removed up front). kind: 'class' | 'reference'."""
    keep, removed = list(DW_ALL), []
    for cls in dw_drop:
        if f"DW_{cls}" in keep:
            keep.remove(f"DW_{cls}"); removed.append((f"DW_{cls}", "class"))
    if dw_reference and f"DW_{dw_reference}" in keep:
        keep.remove(f"DW_{dw_reference}"); removed.append((f"DW_{dw_reference}", "reference"))
    feats = list(BASE_FEATURES) + keep
    assert not set(feats) & set(LEAKY), "LST-derived column leaked into candidate features"
    return feats, removed


def tier_map(tiers):
    return {f: k + 1 for k, group in enumerate(tiers or []) for f in group}


def drop_zero_variance(train, cols):
    kept, log = [], []
    for c in cols:
        if train[c].nunique(dropna=True) <= 1:
            log.append({"feature": c, "reason": "constant in the training split - carries no information"})
        else:
            kept.append(c)
    return kept, log


def spearman_filter(train, cols, threshold=0.8, rule="priority", target=None, tiers=None):
    """Take the most correlated pair with |rho| > threshold, drop ONE member, repeat. Which one (`rule`):
      priority     : lower-priority tier (a-priori, outcome-blind); same tier -> larger mean |rho| to the rest
      mean_abs_rho : larger mean |rho| to the remaining features
      target_corr  : weaker |Spearman rho| with SUHII on train (outcome-dependent)
    Final tie -> alphabetically later is dropped. The triggering pair and the reason are logged."""
    kept, log = list(cols), []
    tm = tier_map(tiers); last = max(tm.values(), default=0) + 1
    tcorr = {c: abs(float(train[c].corr(target, method="spearman"))) for c in kept} if rule == "target_corr" else {}
    while len(kept) > 1:
        rho = np.nan_to_num(train[kept].corr(method="spearman").to_numpy())
        a = np.abs(rho).copy(); np.fill_diagonal(a, 0)
        i, j = np.unravel_index(np.argmax(a), a.shape)
        if a[i, j] <= threshold:
            break
        n, ci, cj = len(kept), kept[i], kept[j]
        mi = (a[i].sum() - a[i, j]) / max(n - 2, 1); mj = (a[j].sum() - a[j, i]) / max(n - 2, 1)
        drop_i, why = None, ""
        if rule == "priority":
            ti, tj = tm.get(ci, last), tm.get(cj, last)
            if ti != tj:
                drop_i = ti > tj
                why = f"{(cj if drop_i else ci)} sits in a higher a-priori priority tier ({min(ti, tj)} vs {max(ti, tj)})"
        elif rule == "target_corr" and tcorr[ci] != tcorr[cj]:
            drop_i = tcorr[ci] < tcorr[cj]
            why = f"{(cj if drop_i else ci)} has the stronger |rho| with SUHII on train ({max(tcorr[ci], tcorr[cj]):.3f} vs {min(tcorr[ci], tcorr[cj]):.3f})"
        if drop_i is None:
            drop_i = mi > mj or (mi == mj and ci > cj)
            why = (f"same priority, so the one more correlated with the other remaining features was removed "
                   f"(mean |rho| {max(mi, mj):.3f} vs {min(mi, mj):.3f})" if rule != "mean_abs_rho" else
                   f"the one more correlated with the other remaining features was removed (mean |rho| {max(mi, mj):.3f} vs {min(mi, mj):.3f})")
        drop, keep = (ci, cj) if drop_i else (cj, ci)
        log.append({"feature": drop, "kept_instead": keep, "pair": f"{ci} ~ {cj}", "rho": round(float(rho[i, j]), 4),
                    "rule": rule, "reason": f"|rho|={abs(rho[i, j]):.3f} with {keep} exceeds {threshold}; {why}."})
        kept.remove(drop)
    return kept, log


def compute_vif(X):
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    n, p = X.shape
    out = np.empty(p)
    for i in range(p):
        y = X[:, i]
        A = np.column_stack([np.ones(n), np.delete(X, i, axis=1)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sst = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - A @ coef) ** 2).sum() / sst if sst > 0 else 1.0
        out[i] = np.inf if r2 >= 1 - 1e-10 else 1 / (1 - r2)
    return out


def vif_filter(train, cols, threshold=8.0, rule="priority", tiers=None):
    """Drop until all VIF <= threshold. priority: among variables over the threshold, remove the one in the lowest
    a-priori tier (ties -> highest VIF). Other rules: highest VIF."""
    kept, log = list(cols), []
    tm = tier_map(tiers); last = max(tm.values(), default=0) + 1
    while len(kept) > 1:
        v = compute_vif(train[kept].to_numpy(float))
        over = [k for k in range(len(kept)) if v[k] > threshold]
        if not over:
            break
        k = max(over, key=lambda q: (tm.get(kept[q], last), v[q])) if rule == "priority" else int(np.argmax(v))
        why = (f"VIF={v[k]:.2f} exceeds {threshold} (linear redundancy with the other kept features); "
               + ("it is the lowest-priority variable among those over the threshold." if rule == "priority" and len(over) > 1
                  else "it is the most redundant variable." if rule != "priority" else "it is the only variable over the threshold."))
        log.append({"feature": kept[k], "vif": float(v[k]), "reason": why})
        kept.pop(k)
    final = pd.DataFrame({"feature": kept, "VIF": compute_vif(train[kept].to_numpy(float)) if kept else []})
    return kept, log, final
