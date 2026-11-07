import json
import numpy as np, pandas as pd, pytest
from uhi.config import load_configs
from uhi.preprocess import filters as F
from uhi.preprocess.clean import clean_rows
from uhi.preprocess.load import load_raw
from uhi.preprocess.pipeline import load_state, preprocess, run_preprocessing
from uhi.preprocess.split import assign_blocks, assign_split, compute_bounds
from uhi.preprocess.transforms import Winsorizer
from tests.fixtures import year_frame, write_years

TIERS = [["NDVI", "NDBI", "DW_built"], ["NDWI", "BSI"], ["NO2", "O3"]]


# ---------- filters ----------
def test_priority_keeps_the_planner_lever_not_the_proxy():
    r = np.random.default_rng(0); x = r.normal(size=500)
    df = pd.DataFrame({"NDVI": x, "NDWI": -x + r.normal(0, .01, 500), "NO2": r.normal(size=500)})
    kept, log = F.spearman_filter(df, list(df), .8, "priority", None, TIERS)
    assert set(kept) == {"NDVI", "NO2"} and log[0]["feature"] == "NDWI" and "tier" in log[0]["reason"]


def test_priority_is_outcome_blind():
    r = np.random.default_rng(1); x = r.normal(size=500)
    df = pd.DataFrame({"NDVI": x, "NDWI": x + r.normal(0, .01, 500)})
    a = F.spearman_filter(df, list(df), .8, "priority", pd.Series(x), TIERS)[0]
    b = F.spearman_filter(df, list(df), .8, "priority", pd.Series(r.normal(size=500)), TIERS)[0]
    assert a == b == ["NDVI"]                      # a different target cannot change the choice


def test_same_tier_falls_back_to_data_rule_and_is_logged():
    r = np.random.default_rng(2); x = r.normal(size=600)
    df = pd.DataFrame({"NDBI": x, "DW_built": x + r.normal(0, .05, 600), "NO2": r.normal(size=600)})
    kept, log = F.spearman_filter(df, list(df), .8, "priority", None, TIERS)
    assert len(kept) == 2 and "same priority" in log[0]["reason"]


def test_vif_handles_perfect_collinearity_and_respects_tiers():
    r = np.random.default_rng(3); a, b = r.normal(size=400), r.normal(size=400)
    df = pd.DataFrame({"NDVI": a, "NO2": b, "O3": a + b})
    kept, log, final = F.vif_filter(df, list(df), 8, "priority", TIERS)
    assert len(kept) == 2 and log[0]["feature"] in ("NO2", "O3") and "NDVI" in kept and (final.VIF <= 8).all()


def test_leakage_guard_and_no_coordinates():
    feats, _ = F.candidate_features(["snow_and_ice", "flooded_vegetation"], "crops")
    assert not {"LST", "LST_ref", "lon", "lat", "NTL", "Pop", "DW_crops"} & set(feats)


# ---------- split ----------
def _frame_with_blocks(years):
    d = pd.concat([year_frame(y, 0) for y in years], ignore_index=True)
    b = compute_bounds(d); d["block_id"] = assign_blocks(d, 10, b)
    return d, b


def test_same_cell_same_split_in_every_year():
    d, _ = _frame_with_blocks([2024, 2025, 2026])
    d["split"], _ = assign_split(d, .7, 42)
    assert (d.groupby("cell_id").split.nunique() == 1).all()


def test_split_is_deterministic_and_close_to_ratio():
    d, _ = _frame_with_blocks([2024])
    s1, m1 = assign_split(d, .7, 42); s2, m2 = assign_split(d, .7, 42)
    assert m1 == m2 and (s1 == s2).all() and 0.6 < (s1 == "train").mean() < 0.8


def test_frozen_split_survives_new_data():
    d, b = _frame_with_blocks([2024])
    _, m1 = assign_split(d, .7, 42)
    d2 = pd.concat([d, d.assign(lon=d.lon + .3)], ignore_index=True)     # new area appears later
    d2["block_id"] = assign_blocks(d2, 10, b)
    _, m2 = assign_split(d2, .7, 42, m1)
    assert all(m2[k] == v for k, v in m1.items())                        # nothing previously seen changed sides


def test_winsor_bounds_come_from_train_only():
    tr = pd.DataFrame({"a": np.arange(100.0)}); te = pd.DataFrame({"a": [1e6, -1e6]})
    w = Winsorizer(.01, .99).fit(tr, ["a"]); o = w.transform(te)
    assert o.a.max() == tr.a.quantile(.99) and o.a.min() == tr.a.quantile(.01)


def test_cleaning_keeps_hot_cells_and_leaves_missing_features_for_the_imputer():
    raw = year_frame(2026, 0)
    raw.loc[3, "NDVI"] = np.nan; raw.loc[4, "Pop"] = -9999; raw.loc[5, "valid_frac"] = .1; raw.loc[6, "SUHII"] = np.nan
    hottest = raw.loc[~raw.index.isin([5, 6]), "SUHII"].max()
    d, log = clean_rows(raw.assign(year=2026))
    assert len(d) == len(raw) - 2                        # only the low valid_frac row and the missing-TARGET row go
    assert d.SUHII.max() == hottest and d.SUHII.isna().sum() == 0
    assert d.NDVI.isna().sum() == 1 and d.Pop.isna().sum() == 1     # features: -9999 became NaN, nothing filled here


# ---------- imputation ----------
def _grid(n=6, year=2026):
    f = year_frame(year, 1, n=n).assign(year=year)
    f["split"] = "train"; f["n_imputed"] = 0
    return f


def test_imputation_copies_the_nearest_cell_in_same_year_and_split():
    from uhi.preprocess.impute import impute_nearest
    f = _grid(); f.loc[7, "NDVI"] = np.nan
    out, log = impute_nearest(f, max_distance_km=50)
    row = out[out.cell_id == f.loc[7, "cell_id"]].iloc[0]
    donors = f[f.NDVI.notna()]
    dist = np.hypot((donors.lon - f.loc[7, "lon"]) * 111.32 * np.cos(np.deg2rad(f.loc[7, "lat"])), (donors.lat - f.loc[7, "lat"]) * 110.57)
    assert row.NDVI == donors.loc[dist.idxmin(), "NDVI"] and row.n_imputed == 1
    assert log["units"]["NDVI"]["imputed"] == 1 and out.NDVI.isna().sum() == 0


def test_imputation_never_borrows_across_the_train_test_boundary():
    from uhi.preprocess.impute import impute_nearest
    f = _grid(); f.loc[:, "split"] = "train"
    f.loc[7, ["NDVI", "split"]] = [np.nan, "test"]
    f.loc[8, "split"] = "test"; f.loc[8, "NDVI"] = 123.0          # the nearest cell sits in the SAME (test) split but far
    out, _ = impute_nearest(f, max_distance_km=500)
    assert out[out.cell_id == f.loc[7, "cell_id"]].iloc[0].NDVI == 123.0
    f.loc[8, "NDVI"] = np.nan                                      # now the test split has NO donor at all
    out2, log2 = impute_nearest(f, max_distance_km=500)
    assert f.loc[7, "cell_id"] not in set(out2.cell_id) and log2["rows_removed_no_donor_in_range"] >= 1


def test_imputation_respects_max_distance_and_removes_instead_of_extrapolating():
    from uhi.preprocess.impute import impute_nearest
    f = _grid(); f.loc[7, "Albedo"] = np.nan
    out, log = impute_nearest(f, max_distance_km=0.001)
    assert f.loc[7, "cell_id"] not in set(out.cell_id) and log["rows_removed_no_donor_in_range"] == 1


def test_dynamic_world_shares_are_copied_together_and_still_sum_to_one():
    from uhi.preprocess.impute import impute_nearest
    from tests.fixtures import DW
    f = _grid(); f.loc[9, ["DW_trees", "DW_built"]] = np.nan
    out, _ = impute_nearest(f, max_distance_km=50)
    r = out[out.cell_id == f.loc[9, "cell_id"]].iloc[0]
    assert abs(r[DW].sum() - 1) < 1e-6 and out[DW].isna().sum().sum() == 0


def test_rows_missing_too_many_features_are_removed_not_repaired():
    from uhi.preprocess.impute import impute_nearest
    f = _grid(); f.loc[7, ["NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL"]] = np.nan
    out, log = impute_nearest(f, max_distance_km=50, max_null_share=0.3)
    assert f.loc[7, "cell_id"] not in set(out.cell_id) and log["rows_removed_too_many_missing"] == 1


def test_a_feature_missing_for_a_whole_year_is_an_error():
    from uhi.preprocess.impute import impute_nearest
    f = _grid(); f["SO2"] = np.nan
    with pytest.raises(ValueError, match="cannot be imputed"):
        impute_nearest(f)


def test_target_is_never_imputed_in_the_pipeline(tmp_path):
    write_years(tmp_path / "raw")
    ref = pd.read_csv(tmp_path / "raw" / "delhi_2025.csv")
    bad = ref.copy(); bad.loc[10:14, "SUHII"] = np.nan; bad.loc[20:24, "NDVI"] = np.nan
    bad.to_csv(tmp_path / "raw" / "delhi_2025.csv", index=False)
    pipe, _ = load_configs()
    res = preprocess(pipe, "delhi", tmp_path / "raw", [2024, 2025, 2026])
    ml = res["ml"]
    assert ml.SUHII.isna().sum() == 0
    assert not set(ref.cell_id[10:15]) & set(ml[ml.year == 2025].cell_id)          # 5 missing-target rows were REMOVED
    assert set(ref.cell_id[20:25]) <= set(ml[ml.year == 2025].cell_id)               # 5 missing-feature rows were KEPT
    assert (ml[ml.year == 2025].set_index("cell_id").loc[list(ref.cell_id[20:25]), "n_imputed"] >= 1).all()
    assert res["impute_log"]["units"]["NDVI"]["imputed"] == 5


def test_imputation_off_falls_back_to_removing_rows(tmp_path):
    write_years(tmp_path / "raw")
    d = pd.read_csv(tmp_path / "raw" / "delhi_2025.csv"); d.loc[20:24, "NDVI"] = np.nan
    d.to_csv(tmp_path / "raw" / "delhi_2025.csv", index=False)
    pipe, _ = load_configs(); pipe["preprocess"]["impute"]["enabled"] = False
    res = preprocess(pipe, "delhi", tmp_path / "raw", [2024, 2025, 2026])
    assert not set(d.cell_id[20:25]) & set(res["ml"][res["ml"].year == 2025].cell_id) and res["impute_log"] is None
    assert [g for g in res["gates"] if g["gate"] == "imputation"][0]["status"] == "SKIP"


# ---------- end to end ----------
@pytest.fixture
def run(tmp_path):
    write_years(tmp_path / "raw")
    pipe, _ = load_configs()
    res = run_preprocessing(pipe, "delhi", tmp_path / "raw", tmp_path / "out", [2024, 2025, 2026], None, [2024, 2025, 2026])
    return pipe, tmp_path, res


def test_only_two_files_are_written(run):
    _, t, res = run
    files = sorted(p.name for p in (t / "out" / "delhi").rglob("*") if p.is_file())
    assert files == ["delhi_ml_ready.parquet", "preprocess_state.json"]


def test_ml_table_is_clean_and_leak_free(run):
    _, t, res = run
    ml = pd.read_parquet(t / "out" / "delhi" / "delhi_ml_ready.parquet")
    assert ml.isna().sum().sum() == 0 and set(ml.year) == {2024, 2025, 2026} and set(ml.split) == {"train", "test"}
    assert not {"LST", "LST_ref", "NTL", "Pop"} & set(ml.columns)
    assert not any(v == "FAIL" for v in res["gates"].values()), res["gates"]


def test_every_column_has_a_documented_decision(run):
    _, t, _ = run
    st = json.loads((t / "out" / "delhi" / "preprocess_state.json").read_text())
    by = {}
    for d in st["decisions"]:
        by.setdefault(d["feature"], []).append(d["action"])
    for c in ["NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL_log", "Pop_log", "SO2", "CO", "NO2", "O3", "AAI",
              "Rain_period", "DW_water", "DW_trees", "DW_grass", "DW_shrub", "DW_built", "DW_bare"]:
        acts = by[c]
        assert acts.count("kept") + acts.count("dropped") == 1, (c, acts)      # exactly one outcome, never lost silently
    assert all(d["reason"] for d in st["decisions"])
    assert "NDVI" in st["final_features"] and "NDWI" not in st["final_features"]  # priority rule picks the lever


def test_second_run_reuses_frozen_split(run):
    pipe, t, res = run
    before = json.loads((t / "out" / "delhi" / "preprocess_state.json").read_text())["split"]["block_map"]
    r2 = run_preprocessing(pipe, "delhi", t / "raw", t / "out", [2024, 2025, 2026], None, [2024, 2025, 2026])
    after = json.loads((t / "out" / "delhi" / "preprocess_state.json").read_text())["split"]["block_map"]
    assert r2["split_reused_from_state"] and before == after


def test_missing_year_file_is_an_error(tmp_path):
    write_years(tmp_path, years=(2024, 2025))
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path, "delhi", [2024, 2025, 2026])


# ---------- year centering / feature spec ----------
def test_year_centering_uses_train_cells_only():
    from uhi.preprocess.transforms import year_center
    d = pd.concat([year_frame(y, 0).assign(year=y) for y in (2024, 2025)], ignore_index=True)
    d["split"] = np.where(d.index % 3 == 0, "test", "train")
    out, m = year_center(d, ["Rain_period"], d.split == "train")
    d2 = d.copy(); d2.loc[d2.split == "test", "Rain_period"] += 1e6              # wreck the test cells
    _, m2 = year_center(d2, ["Rain_period"], d2.split == "train")
    assert m == m2                                                              # test cells cannot move the means
    assert abs(out[(out.split == "train") & (out.year == 2024)].Rain_period.mean()) < 1e-9


def test_state_has_feature_spec_and_year_means(run):
    _, t, _ = run
    st = json.loads((t / "out" / "delhi" / "preprocess_state.json").read_text())
    assert {f["feature"] for f in st["feature_spec"]} == set(st["final_features"])
    assert all("winsor_lower" in f and "transform" in f for f in st["feature_spec"])
    assert set(st["year_centering"]["year_means_train_cells"]) == {"2024", "2025", "2026"}


def test_centred_context_features_carry_no_year_signal(run):
    _, t, _ = run
    ml = pd.read_parquet(t / "out" / "delhi" / "delhi_ml_ready.parquet")
    tr = ml[ml.split == "train"]
    for c in [c for c in ["SO2", "CO", "NO2", "O3", "AAI", "Rain_period"] if c in ml]:
        assert tr.groupby("year")[c].mean().abs().max() < 0.05 * ml[c].std() + 1e-6
