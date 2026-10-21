"""
F1 cost cap paper: data collection and panel construction.

Pulls constructor-level results from the Jolpica-F1 API (Ergast-compatible),
builds a team-season panel, computes competitive balance metrics, and
creates a budget template you fill in by hand from public sources.

Usage:
    pip install requests pandas numpy
    python f1_data_collection.py

Outputs (in ./data):
    races.csv                 one row per race
    constructor_results.csv   one row per constructor per race
    qualifying.csv            one row per driver per qualifying session
    constructor_season.csv    team-season panel (points, share, rank, quali gap)
    balance_metrics.csv       season-level competitive balance metrics
    budgets_template.csv      FILL IN by hand, then rerun to merge
"""

import os
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
START_YEAR = 2005
END_YEAR = 2025          # change to the latest completed season
OUT = "data"
PAGE = 100               # API page size limit
REFETCH = False         # True forces a fresh download instead of using cached raw files
PAUSE = 0.6              # seconds between calls, stay polite to the API

os.makedirs(OUT, exist_ok=True)
session = requests.Session()


def get(url, params=None, retries=5):
    """GET with retry and backoff (handles rate limiting)."""
    for attempt in range(retries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            time.sleep(PAUSE)
            return r.json()
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def fetch_all(path_template, year, table_key, list_key):
    """Page through an endpoint for one season and return the list of records."""
    records, offset = [], 0
    while True:
        url = f"{BASE}/{year}/{path_template}.json"
        data = get(url, {"limit": PAGE, "offset": offset})["MRData"]
        batch = data[table_key][list_key]
        records.extend(batch)
        total = int(data["total"])
        offset += PAGE
        if offset >= total:
            break
    return records


def collect_results(year):
    rows = []
    for race in fetch_all("results", year, "RaceTable", "Races"):
        for res in race["Results"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race": race["raceName"],
                "date": race["date"],
                "constructor_id": res["Constructor"]["constructorId"],
                "constructor": res["Constructor"]["name"],
                "driver_id": res["Driver"]["driverId"],
                "grid": int(res["grid"]),
                "position": int(res["position"]),
                "points": float(res["points"]),
                "status": res["status"],
                "session": "race",
            })
    return rows


def collect_sprints(year):
    """Sprint-race results (points count towards the constructors' standings from 2021)."""
    rows = []
    for race in fetch_all("sprint", year, "RaceTable", "Races"):
        for res in race.get("SprintResults", []):
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race": race["raceName"],
                "date": race["date"],
                "constructor_id": res["Constructor"]["constructorId"],
                "constructor": res["Constructor"]["name"],
                "driver_id": res["Driver"]["driverId"],
                "grid": int(res["grid"]),
                "position": int(res["position"]),
                "points": float(res["points"]),
                "status": res["status"],
                "session": "sprint",
            })
    return rows


def collect_qualifying(year):
    rows = []
    for race in fetch_all("qualifying", year, "RaceTable", "Races"):
        for q in race["QualifyingResults"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "constructor_id": q["Constructor"]["constructorId"],
                "driver_id": q["Driver"]["driverId"],
                "q1": q.get("Q1"), "q2": q.get("Q2"), "q3": q.get("Q3"),
            })
    return rows


def lap_to_seconds(t):
    """Convert 'm:ss.mmm' to seconds; return NaN for missing values."""
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def best_quali_seconds(df):
    """Best available qualifying time per driver-session (prefer latest stage)."""
    for col in ["q1", "q2", "q3"]:
        df[col + "_s"] = df[col].map(lap_to_seconds)
    df["best_quali_s"] = df[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    return df


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return np.nan
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def build_balance_metrics(cs):
    """Season-level competitive balance indicators from constructor points."""
    out = []
    for season, g in cs.groupby("season"):
        g = g.dropna(subset=["points"])      # teams excluded from the official standings
        pts = g["points"].values
        share = g["points_share"].values
        sorted_share = np.sort(share)[::-1]
        out.append({
            "season": season,
            "n_teams": len(g),
            "hhi": float((share ** 2).sum()),
            "gini": float(gini(pts)),
            "top1_share": float(sorted_share[0]),
            "top3_share": float(sorted_share[:3].sum()),
            "champ_minus_median": float(sorted_share[0] - np.median(share)),
            "sd_share": float(np.std(share, ddof=0)),
        })
    return pd.DataFrame(out)


def safe_csv(df, path):
    """Write a CSV; if the file is open in Excel (locked), say so and stop clearly."""
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        raise SystemExit(f"Cannot write {path}: the file is open in another program "
                         f"(probably Excel). Close it and run the script again.")


def main():
    raw_res_path = f"{OUT}/raw_results.csv"
    raw_q_path = f"{OUT}/raw_qualifying.csv"
    if os.path.exists(raw_res_path) and os.path.exists(raw_q_path) and not REFETCH:
        print("Using cached raw data (set REFETCH = True to download again).")
        res = pd.read_csv(raw_res_path)
        quali_raw = pd.read_csv(raw_q_path)
    else:
        all_results, all_quali = [], []
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"Fetching {year} ...")
            all_results.extend(collect_results(year))
            all_quali.extend(collect_qualifying(year))
        res = pd.DataFrame(all_results)
        quali_raw = pd.DataFrame(all_quali)
        safe_csv(res, raw_res_path)       # cache before any mapping
        safe_csv(quali_raw, raw_q_path)
    if "session" not in res.columns:
        res["session"] = "race"          # older cached files only hold races
    res = res[res["session"] == "race"].copy()

    # Sprint points (2021 onward) are NOT in the race-results endpoint, so fetch them separately.
    sprint_path = f"{OUT}/raw_sprint.csv"
    sprint_df = pd.DataFrame()
    if os.path.exists(sprint_path) and not REFETCH:
        sprint_df = pd.read_csv(sprint_path)
        print("Using cached sprint data.")
    else:
        try:
            rows_sp = []
            for year in range(2021, END_YEAR + 1):
                print(f"Fetching sprints {year} ...")
                rows_sp.extend(collect_sprints(year))
            sprint_df = pd.DataFrame(rows_sp)
            if len(sprint_df):
                safe_csv(sprint_df, sprint_path)
        except Exception as err:
            print(f"WARNING: could not fetch sprint results ({err}). Points for 2021 onward will "
                  f"be MISSING sprint points. Fix this before using any points-based result.")
    if len(sprint_df):
        res = pd.concat([res, sprint_df], ignore_index=True)
    quali = best_quali_seconds(quali_raw)

    # Rebranded teams (e.g. Force India -> Racing Point -> Aston Martin) must be
    # ONE team in the panel. data/team_map.csv maps each Jolpica constructor_id
    # to a stable team_id. Ids not listed in the map keep their own id.
    map_path = f"{OUT}/team_map.csv"
    if os.path.exists(map_path):
        try:
            tm = pd.read_csv(map_path).set_index("constructor_id")["team_id"].to_dict()
        except (pd.errors.EmptyDataError, KeyError):
            raise SystemExit("data/team_map.csv is empty or has the wrong columns. It must have a "
                             "header row 'constructor_id,team_id' followed by one row per id. "
                             "Replace it with the team_map.csv supplied with the project.")
    else:
        tm = {}
        print("No data/team_map.csv found: rebranded teams will be treated as "
              "separate teams. Copy team_map.csv into ./data and rerun.")
    for df_ in (res, quali):
        df_["constructor_id_raw"] = df_["constructor_id"]
        df_["constructor_id"] = df_["constructor_id"].map(tm).fillna(df_["constructor_id"])
    print("Constructor ids in data:", sorted(res["constructor_id_raw"].unique()))

    safe_csv(res, f"{OUT}/constructor_results.csv")
    safe_csv(quali, f"{OUT}/qualifying.csv")
    safe_csv(res[["season", "round", "race", "date"]].drop_duplicates(), f"{OUT}/races.csv")

    # Team-season points, share and rank. Points = race + sprint. Races and wins use race rows only.
    # Group by season and team id only: a team renamed mid-season (Force India -> Racing Point
    # in 2018) must stay ONE row. The name kept is the latest one in that season.
    keys = ["season", "constructor_id"]
    race_rows = res[res["session"] == "race"].sort_values(["season", "round"])
    cs = (race_rows.groupby(keys, as_index=False)
                   .agg(constructor=("constructor", "last"),
                        races=("round", "nunique"),
                        wins=("position", lambda s_: int((s_ == 1).sum())),
                        constructor_ids_raw=("constructor_id_raw",
                                             lambda s_: ",".join(sorted(set(s_))))))
    pts = res.groupby(keys, as_index=False).agg(points=("points", "sum"))
    sp = (res[res["session"] == "sprint"].groupby(keys, as_index=False)
             .agg(sprint_points=("points", "sum")))
    cs = cs.merge(pts, on=keys, how="left").merge(sp, on=keys, how="left")
    cs["sprint_points"] = cs["sprint_points"].fillna(0.0)   # no sprint points scored = true zero
    cs["points_race_sum"] = cs["points"]

    # Documented adjustment: official standings differ from summed race points in one case.
    # 2018: points scored by the original Force India entry were removed from the official
    # standings; the continuing Racing Point Force India entry is credited with 52 points.
    # 2020: Racing Point received a 15-point constructors' penalty (brake-duct case), so the
    #       official total is 195, not the 210 that summing driver points gives.
    # 2007: McLaren was excluded from the constructors' championship (NaN = not ranked).
    OFFICIAL_POINTS = {(2018, "aston_martin"): 52.0,
                       (2020, "aston_martin"): 195.0,
                       (2007, "mclaren"): float("nan")}
    for (yr_, tid_), val_ in OFFICIAL_POINTS.items():
        m_ = (cs["season"] == yr_) & (cs["constructor_id"] == tid_)
        if m_.any():
            shown = "excluded from the standings" if pd.isna(val_) else f"set to official {val_}"
            print(f"NOTE: {tid_} {yr_} points {shown} "
                  f"(summed race points were {cs.loc[m_, 'points'].iloc[0]}).")
            cs.loc[m_, "points"] = val_

    cs["points_share"] = cs["points"] / cs.groupby("season")["points"].transform("sum")
    cs["rank"] = cs.groupby("season")["points"].rank(ascending=False, method="min")

    # Season totals as a check against the official standings (compare by hand)
    safe_csv(cs[["season", "constructor_id", "constructor_ids_raw", "points_race_sum",
                 "sprint_points", "points", "rank"]].sort_values(["season", "rank"]),
             f"{OUT}/points_check.csv")

    # Qualifying gap: team's average best time vs the fastest team, per race
    # Qualifying gap per DRIVER first (each driver's best lap vs the race's pole lap), then per team:
    #   quali_gap_pct       = the team's FASTEST car (car-performance ceiling; less noise from
    #                         one driver's crash, penalty or mechanical failure). PRIMARY measure.
    #   quali_gap_mean_pct  = AVERAGE of the team's timed cars (reflects both cars, but noisier).
    # Both are then averaged over the season's races. Drivers with no time are left out.
    dq = quali.dropna(subset=["best_quali_s"]).copy()
    pole = dq.groupby(["season", "round"])["best_quali_s"].transform("min")
    dq["gap_pct"] = 100 * (dq["best_quali_s"] - pole) / pole
    tq = (dq.groupby(["season", "round", "constructor_id"], as_index=False)
            .agg(gap_best=("gap_pct", "min"), gap_mean=("gap_pct", "mean"),
                 n_timed=("gap_pct", "size")))
    qgap = (tq.groupby(["season", "constructor_id"], as_index=False)
              .agg(quali_gap_pct=("gap_best", "mean"),
                   quali_gap_mean_pct=("gap_mean", "mean"),
                   races_timed=("n_timed", "size"),
                   races_both_timed=("n_timed", lambda s_: int((s_ >= 2).sum()))))
    cs = cs.merge(qgap, on=["season", "constructor_id"], how="left")

    # Lagged rank for the persistence model
    cs = cs.sort_values(["constructor_id", "season"])
    cs["rank_lag"] = cs.groupby("constructor_id")["rank"].shift(1)

    # Regulation-era and treatment indicators
    cs["post2021"] = (cs["season"] >= 2021).astype(int)
    cs["post2022"] = (cs["season"] >= 2022).astype(int)
    cs["reset_2009"] = (cs["season"] == 2009).astype(int)
    cs["reset_2014"] = (cs["season"] == 2014).astype(int)
    cs["reset_2022"] = (cs["season"] == 2022).astype(int)
    cs["covid_2020"] = (cs["season"] == 2020).astype(int)

    # Budget template: fill in manually, then rerun to merge
    budget_path = f"{OUT}/budgets.csv"
    template = cs[["season", "constructor_id", "constructor"]].copy()
    template["budget_usd_m"] = np.nan
    template["source"] = ""
    if not os.path.exists(f"{OUT}/budgets_template.csv"):
        safe_csv(template, f"{OUT}/budgets_template.csv")

    if os.path.exists(budget_path):
        b = pd.read_csv(budget_path)[["season", "constructor_id", "budget_usd_m"]]
        cs = cs.merge(b, on=["season", "constructor_id"], how="left")
        # Mean of the budget values that EXIST for 2018-2020 (missing years are ignored, not
        # treated as zero). With RaceFans data this is a two-year (2018-2019) average.
        pre = (cs[cs["season"].between(2018, 2020)]
               .groupby("constructor_id")["budget_usd_m"].mean()
               .rename("pre_cap_budget"))
        yrs_used = sorted(cs.loc[cs["season"].between(2018, 2020) & cs["budget_usd_m"].notna(),
                                 "season"].unique().astype(int).tolist())
        print("Pre-cap budget average uses these seasons:", yrs_used)
        cs = cs.merge(pre, on="constructor_id", how="left")
        # Standardise across TEAMS (one value per team), not across team-seasons
        team_pre = cs.groupby("constructor_id")["pre_cap_budget"].first().dropna()
        cs["pre_cap_budget_z"] = (
            (cs["pre_cap_budget"] - team_pre.mean()) / team_pre.std()
        )
        cs["points_per_m"] = cs["points"] / cs["budget_usd_m"]
        print("Merged budgets from budgets.csv")
    else:
        print("No budgets.csv found. Copy budgets_template.csv to budgets.csv, "
              "fill budget_usd_m and source, then rerun.")

    safe_csv(cs, f"{OUT}/constructor_season.csv")
    safe_csv(build_balance_metrics(cs), f"{OUT}/balance_metrics.csv")
    print(f"Done. Files written to ./{OUT}")


if __name__ == "__main__":
    main()