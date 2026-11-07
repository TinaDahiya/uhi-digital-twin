"""Stage 0: city configuration + study-period resolution."""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(pipeline_path=None, cities_path=None) -> tuple[dict, dict]:
    pipe = load_yaml(pipeline_path or ROOT / "configs" / "pipeline.yaml")
    cities = load_yaml(cities_path or ROOT / "configs" / "cities.yaml")["cities"]
    return pipe, cities


def get_city(cities: dict, key: str) -> dict:
    if key not in cities:
        raise KeyError(f"Unknown city '{key}'. Known: {sorted(cities)}")
    return cities[key]


def utm_epsg(city_cfg: dict) -> int:
    """UTM (north) EPSG code derived from the city's centre longitude."""
    lon = city_cfg["center"]["lon"]
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone


def season_dates(city_cfg: dict, year: int) -> tuple[date, date]:
    """(start, end_exclusive) of the city's peak-heat window."""
    s = city_cfg.get("season")
    if not s:
        raise ValueError(
            f"{city_cfg.get('display_name')} has no validated season window; "
            "confirm against IMD normals and set `season` in configs/cities.yaml."
        )
    start = date(year, s["start_month"], 1)
    last_day = calendar.monthrange(year, s["end_month"])[1]
    end_excl = date(year, s["end_month"], last_day) + timedelta(days=1)
    return start, end_excl


def tropomi_dates(city_cfg: dict, year: int) -> tuple[date, date]:
    """(start, end_exclusive) of the narrow TROPOMI window; must sit inside the season."""
    tw = city_cfg.get("tropomi_window")
    if not tw:
        raise ValueError("No tropomi_window configured for this city.")
    m, d = (int(x) for x in tw["start_md"].split("-"))
    start = date(year, m, d)
    end_excl = start + timedelta(days=int(tw["length_days"]))
    s0, s1 = season_dates(city_cfg, year)
    if not (s0 <= start and end_excl <= s1):
        raise ValueError(f"TROPOMI window {start}..{end_excl} falls outside season {s0}..{s1}")
    return start, end_excl


def resolve_latest_year(city_cfg: dict, buffer_days: int = 30, min_year: int = 2019,
                        today: date | None = None) -> int:
    """Most recent year whose season ended at least `buffer_days` ago."""
    today = today or date.today()
    for y in range(today.year, min_year - 1, -1):
        _, end_excl = season_dates(city_cfg, y)
        if end_excl + timedelta(days=buffer_days) <= today:
            return y
    raise RuntimeError("No complete season available on/after min_year.")


def month_floor(d: date) -> date:
    return date(d.year, d.month, 1)


def months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def is_partial_year(year: int, today: date | None = None, buffer_days: int = 30) -> bool:
    """A year is 'partial' while its end (+ a data-latency buffer) has not passed."""
    today = today or date.today()
    return date(year + 1, 1, 1) + timedelta(days=buffer_days) > today


def resolve_years(pipe: dict, city_cfg: dict, today: date | None = None) -> list[int]:
    """Study years, ascending. `years: [..]` wins; otherwise the last `n_years` (season mode: complete seasons;
    full_year mode: calendar years ending with the current, possibly partial, year)."""
    ys = pipe.get("years")
    if ys:
        return sorted(int(y) for y in ys)
    today = today or date.today()
    n, miny = int(pipe.get("n_years", 3)), pipe.get("min_year", 2019)
    if pipe.get("window", "full_year") == "season":
        latest = resolve_latest_year(city_cfg, pipe.get("latest_year_buffer_days", 30), miny, today)
    else:   # include the current year only if it already has enough complete months
        latest = today.year if today.month - 1 >= pipe.get("min_partial_months", 3) else today.year - 1
    years = list(range(latest - n + 1, latest + 1))
    if years[0] < miny:
        raise ValueError(f"{n} years ending {latest} start before min_year={miny}")
    return years


def plan_periods(pipe: dict, city_cfg: dict, years: list[int], today: date | None = None,
                 covered_through: date | None = None) -> dict[int, dict]:
    """Exact compositing window of each year (end is exclusive).

    window: season    -> the city's peak-heat season (the original design).
    window: full_year -> 1 Jan .. 1 Jan of next year. A PARTIAL year (still running) ends at the first day of the month
                         after the last complete month that every source covers (`covered_through`).
    align_months: true -> every year is trimmed to the same calendar months as the partial year, so composites are
                         like-for-like (avoids seasonal bias between a full and a partial year).
    TROPOMI: mode `period` -> same window as the period; mode `narrow` -> the city's 14-day window.
    """
    today = today or date.today()
    out: dict[int, dict] = {}
    if pipe.get("window", "full_year") == "season":
        for y in years:
            s0, s1 = season_dates(city_cfg, y)
            t = tropomi_dates(city_cfg, y) if pipe.get("tropomi", {}).get("mode", "period") == "narrow" else (s0, s1)
            out[y] = {"start": s0, "end_excl": s1, "months": months_between(s0, s1), "partial": False, "tropomi": t}
        return out
    buf, min_m = pipe.get("latest_year_buffer_days", 30), pipe.get("min_partial_months", 3)
    for y in years:
        start, end = date(y, 1, 1), date(y + 1, 1, 1)
        partial = is_partial_year(y, today, buf)
        if partial:
            end = min(end, month_floor(covered_through) if covered_through else month_floor(today))
            if months_between(start, end) < min_m:
                raise ValueError(f"{y}: only {months_between(start, end)} complete month(s) of data available "
                                 f"(need >= {min_m}); drop {y} from `years` or wait.")
        out[y] = {"start": start, "end_excl": end, "partial": partial}
    if pipe.get("align_months"):
        cut = min((p["end_excl"].month for p in out.values() if p["partial"]), default=None)
        if cut:
            for y, p in out.items():
                p["end_excl"] = min(p["end_excl"], date(y, cut, 1))
    for y, p in out.items():
        p["months"] = months_between(p["start"], p["end_excl"])
        if pipe.get("tropomi", {}).get("mode", "period") == "narrow":
            p["tropomi"] = tropomi_dates(city_cfg, y)
            if not (p["start"] <= p["tropomi"][0] and p["tropomi"][1] <= p["end_excl"]):
                raise ValueError(f"{y}: narrow TROPOMI window falls outside the period")
        else:
            p["tropomi"] = (p["start"], p["end_excl"])
    return out
