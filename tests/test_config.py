from datetime import date
import pytest
from uhi.config import (load_configs, plan_periods, resolve_latest_year, resolve_years, season_dates,
                        tropomi_dates, utm_epsg, is_partial_year)
from uhi.gee.availability import covered_through


def _p(**kw):
    p, c = load_configs()
    return {**p, **kw}, c


def test_utm_derived_from_lon():
    _, c = load_configs()
    assert utm_epsg(c["delhi"]) == 32643 and utm_epsg(c["hyderabad"]) == 32644


def test_explicit_years_win_and_default_is_2024_2026():
    p, c = _p()
    assert resolve_years(p, c["delhi"]) == [2024, 2025, 2026]
    assert resolve_years({**p, "years": [2025, 2024]}, c["delhi"]) == [2024, 2025]


def test_year_resolution_when_not_explicit():
    p, c = _p(years=None)
    assert resolve_years(p, c["delhi"], today=date(2026, 11, 3)) == [2024, 2025, 2026]
    assert resolve_years(p, c["delhi"], today=date(2027, 2, 1)) == [2024, 2025, 2026]   # 2027 has < 3 complete months
    assert resolve_years({**p, "window": "season"}, c["delhi"], today=date(2026, 6, 10)) == [2023, 2024, 2025]
    with pytest.raises(ValueError):
        resolve_years({**p, "n_years": 9}, c["delhi"], today=date(2026, 11, 3))


def test_partial_year_detection():
    assert is_partial_year(2026, date(2026, 11, 3)) and is_partial_year(2026, date(2027, 1, 10))
    assert not is_partial_year(2025, date(2026, 11, 3))


def test_full_years_plus_partial_year_ends_at_latest_complete_month():
    p, c = _p()
    plan = plan_periods(p, c["delhi"], [2024, 2025, 2026], today=date(2026, 11, 7), covered_through=date(2026, 9, 18))
    assert (plan[2024]["start"], plan[2024]["end_excl"], plan[2024]["months"]) == (date(2024, 1, 1), date(2025, 1, 1), 12)
    assert (plan[2025]["end_excl"], plan[2025]["months"], plan[2025]["partial"]) == (date(2026, 1, 1), 12, False)
    assert (plan[2026]["end_excl"], plan[2026]["months"], plan[2026]["partial"]) == (date(2026, 9, 1), 8, True)  # Sep not complete
    assert plan[2026]["tropomi"] == (date(2026, 1, 1), date(2026, 9, 1))


def test_align_months_trims_every_year_to_the_same_months():
    p, c = _p(align_months=True)
    plan = plan_periods(p, c["delhi"], [2024, 2025, 2026], today=date(2026, 11, 7), covered_through=date(2026, 10, 5))
    assert [v["months"] for v in plan.values()] == [9, 9, 9]
    assert plan[2024]["end_excl"] == date(2024, 10, 1)


def test_partial_year_with_too_little_data_is_refused():
    p, c = _p()
    with pytest.raises(ValueError):
        plan_periods(p, c["delhi"], [2026], today=date(2026, 11, 7), covered_through=date(2026, 2, 20))


def test_season_mode_keeps_the_original_design():
    p, c = _p(window="season", tropomi={"mode": "narrow"})
    plan = plan_periods(p, c["delhi"], [2026])
    assert (plan[2026]["start"], plan[2026]["end_excl"]) == season_dates(c["delhi"], 2026)
    assert plan[2026]["tropomi"] == tropomi_dates(c["delhi"], 2026)


def test_unvalidated_city_season_refused():
    p, c = _p(window="season")
    with pytest.raises(ValueError):
        plan_periods(p, c["hyderabad"], [2026])


def test_covered_through_finds_the_binding_source():
    latest = {"Landsat 8 L2": date(2026, 11, 2), "VIIRS monthly": date(2026, 8, 1), "CHIRPS daily": date(2026, 10, 30)}
    d, binding, per = covered_through(latest)
    assert binding == "VIIRS monthly" and d == date(2026, 9, 1)     # a monthly composite covers its whole month
