"""
F1 cost cap paper: analysis.

Run f1_data_collection.py first and make sure data/budgets.csv is filled in.

    pip install pandas numpy statsmodels linearmodels matplotlib scipy
    python f1_analysis.py

Outputs go to ./results (tables as CSV/TXT, figures as PNG and PDF).

Notes
-----
* PreCapBudget is a team-level constant: the mean of the budget values that exist for
  2018-2020. With RaceFans estimates only 2018 and 2019 exist, so in practice it is a
  TWO-YEAR (2018-2019) average. It is a proxy for pre-cap resources, not cap-basis spending.
* Because it is constant within a team, its main effect is absorbed by team fixed effects.
  Only the interaction with the post period is identified. Season fixed effects absorb
  anything common to all teams, so this model estimates whether higher-budget teams CHANGED
  DIFFERENTLY after 2021. It does NOT estimate the overall effect of the cap on the whole
  field (the season-level time-series tables address that separately).
* Event-study coefficients are relative to the base year (2020, a COVID season), not
  stand-alone annual effects.
* With ~10-15 teams, conventional clustered SEs are unreliable. The script
  also reports permutation p-values (budget labels shuffled across teams).
* Nothing here is pre-filled. Every number comes from your data.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from scipy import stats

warnings.filterwarnings("ignore")

DATA = "data"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ---- Settings you may want to change -------------------------------------
SAMPLE_START = 2005          # balance metrics sample
DID_START = 2014             # DiD / event-study window start
DROP_COVID = False           # True drops 2020 from regression samples
N_PERM = 999                 # permutation replications
CAP_USD = 145                # cost cap level used for the 'excess over cap' checks ($m)
# Primary outcome: season-average qualifying gap to pole (%), lower = closer to the fastest car.
# Secondary outcome: constructor share of season points.
OUTCOME = "quali_gap_pct"
ALT_OUTCOME = "points_share"
# SIGN GUIDE. Quali gap: convergence after the cap means high-budget teams get RELATIVELY SLOWER,
# so the budget x post coefficient is expected to be POSITIVE (H2). Points share: expected NEGATIVE.
SEED = 42
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

# Restrained, publication-style figure defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
})
NAVY, GREY, RED = "#1F3A5F", "#8A8A8A", "#B03A2E"


def save_fig(fig, name):
    from matplotlib.ticker import MaxNLocator
    for _ax in fig.axes:
        _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


def mark_events(ax):
    for yr, label in [(2014, "2014 PU reset"), (2021, "Cost cap"),
                      (2022, "2022 aero reset")]:
        ax.axvline(yr, color=GREY, linestyle="--", linewidth=0.9)
        ax.text(yr + 0.1, ax.get_ylim()[1], label, rotation=90, va="top",
                ha="left", fontsize=8, color=GREY)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cs = pd.read_csv(f"{DATA}/constructor_season.csv")
bm = pd.read_csv(f"{DATA}/balance_metrics.csv")
bm = bm[bm["season"] >= SAMPLE_START].copy()
# Normalised measures adjust for the number of teams (the grid size changed over time):
#   HHI*  = (HHI - 1/N) / (1 - 1/N)   (0 = perfectly equal, 1 = one team has everything)
#   Gini* = Gini * N / (N - 1)        (0 = equal, 1 = maximally unequal)
bm["hhi_norm"] = (bm["hhi"] - 1 / bm["n_teams"]) / (1 - 1 / bm["n_teams"])
bm["gini_norm"] = bm["gini"] * bm["n_teams"] / (bm["n_teams"] - 1)

# Guard: one row per team-season (an older data script could create two rows in 2018)
if cs.duplicated(["season", "constructor_id"]).any():
    raise SystemExit("constructor_season.csv has duplicate team-seasons. Download the latest "
                     "f1_data_collection.py, rerun it, then rerun this script.")

# ---- Coverage report: which team-seasons have which data (blanks stay blank, never zero) ----
cov_rows = []
for t, g in cs.groupby("constructor_id"):
    bud_years = sorted(g.loc[g["budget_usd_m"].notna(), "season"].astype(int).tolist()) \
        if "budget_usd_m" in g.columns else []
    cov_rows.append({
        "team_id": t,
        "first_season": int(g["season"].min()), "last_season": int(g["season"].max()),
        "n_seasons": int(g["season"].nunique()),
        "n_with_qualifying_gap": int(g["quali_gap_pct"].notna().sum()),
        "budget_years": " ".join(map(str, bud_years)),
        "has_pre_cap_budget": bool("pre_cap_budget_z" in g.columns and g["pre_cap_budget_z"].notna().any()),
    })
cov = pd.DataFrame(cov_rows)
cov.to_csv(f"{OUT}/table0_coverage.csv", index=False)
no_budget = cov.loc[(~cov["has_pre_cap_budget"]) & (cov["last_season"] >= 2014), "team_id"].tolist()
print("Teams since 2014 WITHOUT a pre-cap budget (excluded from the DiD):", no_budget)

if "pre_cap_budget_z" not in cs.columns:
    raise SystemExit("pre_cap_budget_z missing. Fill data/budgets.csv and rerun "
                     "f1_data_collection.py first.")

# ---------------------------------------------------------------------------
# 1. Competitive balance: descriptive figure and pre/post comparison
# ---------------------------------------------------------------------------
metrics = ["hhi", "hhi_norm", "gini", "gini_norm", "top3_share", "champ_minus_median"]
labels = {
    "hhi": "HHI of constructor points",
    "hhi_norm": "Normalised HHI (adjusts for number of teams)",
    "gini_norm": "Normalised Gini (adjusts for number of teams)",
    "gini": "Gini of constructor points",
    "top3_share": "Top-3 points share",
    "champ_minus_median": "Champion minus median share",
}

fig, axes = plt.subplots(3, 2, figsize=(9, 8.5), sharex=True)
for ax, m in zip(axes.ravel(), metrics):
    ax.plot(bm["season"], bm[m], color=NAVY, marker="o", markersize=3.5,
            linewidth=1.4)
    ax.set_title(labels[m])
    mark_events(ax)
for ax in axes[-1]:
    ax.set_xlabel("Season")
save_fig(fig, "fig1_balance_metrics")

rows = []
for m in metrics:
    pre = bm.loc[bm["season"] < 2021, m].dropna()
    post = bm.loc[bm["season"] >= 2021, m].dropna()
    t, p = stats.ttest_ind(post, pre, equal_var=False)
    u, pu = stats.mannwhitneyu(post, pre, alternative="two-sided")
    rows.append({"metric": m, "pre_mean": pre.mean(), "post_mean": post.mean(),
                 "diff": post.mean() - pre.mean(), "n_pre": len(pre),
                 "n_post": len(post), "welch_p": p, "mannwhitney_p": pu})
tab2 = pd.DataFrame(rows)
tab2.to_csv(f"{OUT}/table2_pre_post_balance.csv", index=False)

# Inter-season dimension: Spearman rank correlation between consecutive seasons' standings.
# Values near 1 mean the order barely changes (a "frozen" hierarchy); lower = more reshuffling.
pts_wide = (cs.dropna(subset=["points"])
              .pivot_table(index="season", columns="constructor_id", values="points"))
rho_rows = []
for season_ in sorted(pts_wide.index):
    prev_ = season_ - 1
    if prev_ not in pts_wide.index or season_ < SAMPLE_START + 1:
        continue
    pair = pts_wide.loc[[prev_, season_]].T.dropna()      # teams present in both seasons
    if len(pair) >= 5:
        rho_, p_ = stats.spearmanr(pair[prev_], pair[season_])
        rho_rows.append({"season": season_, "spearman_rho": rho_, "n_teams_both_seasons": len(pair)})
srcc = pd.DataFrame(rho_rows)
srcc.to_csv(f"{OUT}/table2i_interseason_spearman.csv", index=False)
pre_r = srcc.loc[srcc["season"] < 2021, "spearman_rho"]
post_r = srcc.loc[srcc["season"] >= 2021, "spearman_rho"]
# Two windows: all seasons, and 2014 onward. Before 2014 several teams changed identity and are not
# merged into one lineage, so fewer teams overlap between seasons; the 2014+ window avoids that.
sp_rows = []
for label_, lo_ in [("all seasons", SAMPLE_START + 1), ("2014 onward", 2014)]:
    a_ = srcc.loc[(srcc["season"] >= lo_) & (srcc["season"] < 2021), "spearman_rho"]
    if len(a_) > 2 and len(post_r) > 2:
        sp_rows.append({"window": label_, "pre_mean": a_.mean(), "post_mean": post_r.mean(),
                        "n_pre": len(a_), "n_post": len(post_r),
                        "welch_p": stats.ttest_ind(post_r, a_, equal_var=False)[1],
                        "mannwhitney_p": stats.mannwhitneyu(post_r, a_, alternative="two-sided")[1]})
pd.DataFrame(sp_rows).to_csv(f"{OUT}/table2j_spearman_pre_post.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(srcc["season"], srcc["spearman_rho"], color=NAVY, marker="o", markersize=3.5, linewidth=1.4)
ax.set_xlabel("Season")
ax.set_ylabel("Spearman rank correlation with previous season")
ax.set_title("Inter-season persistence of the constructors' order")
mark_events(ax)
save_fig(fig, "fig1c_interseason_spearman")

# Interrupted time series with reset dummies (HAC SEs)
bm["trend"] = bm["season"] - SAMPLE_START
bm["post2021"] = (bm["season"] >= 2021).astype(int)
bm["reset_2009"] = (bm["season"] == 2009).astype(int)
bm["reset_2014"] = (bm["season"] == 2014).astype(int)
bm["reset_2022"] = (bm["season"] == 2022).astype(int)

its_out = []
for m in metrics:
    mod = smf.ols(f"{m} ~ trend + post2021 + reset_2009 + reset_2014 + reset_2022",
                  data=bm).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    its_out.append(f"=== ITS: {m} ===\n{mod.summary().tables[1]}\n")
open(f"{OUT}/table2b_interrupted_time_series.txt", "w").write("\n".join(its_out))

# ---------------------------------------------------------------------------
# Season-level qualifying gap: mean and spread across teams
# ---------------------------------------------------------------------------
qg = (cs[cs["season"] >= SAMPLE_START].dropna(subset=["quali_gap_pct"])
        .groupby("season")["quali_gap_pct"].agg(["mean", "median", "std"]).reset_index())
qg.to_csv(f"{OUT}/season_quali_gap.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(qg["season"], qg["mean"], color=NAVY, marker="o", markersize=3.5, linewidth=1.4)
ax.set_xlabel("Season")
ax.set_ylabel("Mean qualifying gap to pole (%)")
ax.set_title("Season-average qualifying gap across teams")
mark_events(ax)
save_fig(fig, "fig1b_quali_gap")
pre_q = qg.loc[qg["season"] < 2021, "mean"]
post_q = qg.loc[qg["season"] >= 2021, "mean"]
pd.DataFrame([{"pre_mean": pre_q.mean(), "post_mean": post_q.mean(),
               "n_pre": len(pre_q), "n_post": len(post_q),
               "welch_p": stats.ttest_ind(post_q, pre_q, equal_var=False)[1]}]
             ).to_csv(f"{OUT}/table2c_quali_gap_pre_post.csv", index=False)

# Interrupted time series on the season-level gap: does the post-2021 level sit below
# the pre-existing trend? (2014+ window avoids the 2010-12 new-team distortion.)
qg["trend"] = qg["season"] - qg["season"].min()
qg["post2021"] = (qg["season"] >= 2021).astype(int)
qg["reset_2022"] = (qg["season"] == 2022).astype(int)
qg["reset_2014"] = (qg["season"] == 2014).astype(int)
its_rows = []
for label, sub in [("2014 onward", qg[qg["season"] >= 2014]), ("all seasons", qg)]:
    for yv in ["mean", "median", "std"]:
        f = f"{yv} ~ trend + post2021 + reset_2022" + (" + reset_2014" if label == "all seasons" else "")
        m_ = smf.ols(f, data=sub).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        its_rows.append({"window": label, "outcome": yv,
                         "post2021_level_shift": m_.params["post2021"],
                         "se": m_.bse["post2021"], "p": m_.pvalues["post2021"],
                         "trend_per_year": m_.params["trend"], "p_trend": m_.pvalues["trend"],
                         "n": int(m_.nobs)})
pd.DataFrame(its_rows).to_csv(f"{OUT}/table2d_quali_gap_its.csv", index=False)

# ---------------------------------------------------------------------------
# Regression sample
# ---------------------------------------------------------------------------
d = cs[(cs["season"] >= DID_START)].dropna(
    subset=[OUTCOME, "pre_cap_budget_z"]).copy()
if DROP_COVID:
    d = d[d["season"] != 2020]
d["cap_x_post"] = d["pre_cap_budget_z"] * d["post2021"]
d["cap_x_2022"] = d["pre_cap_budget_z"] * (d["season"] == 2022).astype(int)
d = d.set_index(["constructor_id", "season"]).sort_index()
print(f"DiD sample: {d.shape[0]} team-seasons, "
      f"{d.index.get_level_values(0).nunique()} teams")


def fit_fe(df, y, xcols, cluster=True):
    mod = PanelOLS(df[y], df[xcols], entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    return mod.fit(cov_type="clustered", cluster_entity=True) if cluster \
        else mod.fit()


# ---------------------------------------------------------------------------
# 2. Main DiD with treatment intensity
# ---------------------------------------------------------------------------
specs = {
    "(1) Baseline": ["cap_x_post"],
    "(2) + 2022 reset interaction": ["cap_x_post", "cap_x_2022"],
}
did_rows, did_text = [], []
for name, xs in specs.items():
    res = fit_fe(d, OUTCOME, xs)
    did_text.append(f"=== {name} ===\n{res.summary.tables[1]}\n")
    for x in xs:
        did_rows.append({"spec": name, "term": x, "coef": res.params[x],
                         "se": res.std_errors[x], "p_cluster": res.pvalues[x],
                         "n": int(res.nobs)})
tab3 = pd.DataFrame(did_rows)

# Permutation inference for the baseline coefficient
obs = fit_fe(d, OUTCOME, ["cap_x_post"]).params["cap_x_post"]
teams = d.index.get_level_values(0).unique()
budget_by_team = d.groupby(level=0)["pre_cap_budget_z"].first()
perm = []
for _ in range(N_PERM):
    shuffled = pd.Series(rng.permutation(budget_by_team.values),
                         index=budget_by_team.index)
    tmp = d.copy()
    tmp["pre_cap_budget_z"] = tmp.index.get_level_values(0).map(shuffled)
    tmp["cap_x_post"] = tmp["pre_cap_budget_z"] * tmp["post2021"]
    perm.append(fit_fe(tmp, OUTCOME, ["cap_x_post"],
                       cluster=False).params["cap_x_post"])
perm = np.array(perm)
p_perm = (1 + (np.abs(perm) >= abs(obs)).sum()) / (N_PERM + 1)
tab3.loc[tab3["term"].eq("cap_x_post") & tab3["spec"].eq("(1) Baseline"),
         "p_permutation"] = p_perm
tab3.to_csv(f"{OUT}/table3_did.csv", index=False)
open(f"{OUT}/table3_did_full.txt", "w").write("\n".join(did_text))
print(f"Baseline DiD coef = {obs:.4f}, permutation p = {p_perm:.3f}")

# ---------------------------------------------------------------------------
# 3. Event study (base year = 2020)
# ---------------------------------------------------------------------------
BASE_YEAR = 2020
years = sorted(set(d.index.get_level_values(1)) - {BASE_YEAR})
ev = d.copy()
for y in years:
    ev[f"b_{y}"] = ev["pre_cap_budget_z"] * (ev.index.get_level_values(1) == y)
ev_x = [f"b_{y}" for y in years]
ev_res = fit_fe(ev, OUTCOME, ev_x)
# Joint test that all pre-period event-study coefficients are zero (parallel-trends check).
# With ~10 clusters this chi-square test is only indicative; also read the plot.
pre_cols = [f"b_{y}" for y in years if y < 2020]
try:
    names = list(ev_res.params.index)
    R = np.zeros((len(pre_cols), len(names)))
    for i, c_ in enumerate(pre_cols):
        R[i, names.index(c_)] = 1.0
    w_ = ev_res.wald_test(restriction=R, value=np.zeros(len(pre_cols)))
    pd.DataFrame([{"test": "joint pre-trend (all pre-2020 coefficients = 0)",
                   "wald_stat": float(w_.stat), "df": len(pre_cols), "p": float(w_.pval),
                   "n_clusters": int(d.index.get_level_values(0).nunique())}]
                 ).to_csv(f"{OUT}/table8_pretrend_test.csv", index=False)
    print(f"Joint pre-trend test: stat = {float(w_.stat):.2f}, p = {float(w_.pval):.3f}")
except Exception as err:
    print(f"Pre-trend test skipped: {err}")

coefs = pd.DataFrame({
    "year": years,
    "coef": [ev_res.params[f"b_{y}"] for y in years],
    "se": [ev_res.std_errors[f"b_{y}"] for y in years],
})
coefs["lo"] = coefs["coef"] - 1.96 * coefs["se"]
coefs["hi"] = coefs["coef"] + 1.96 * coefs["se"]
coefs.to_csv(f"{OUT}/event_study_coefs.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(coefs["year"], coefs["coef"],
            yerr=[coefs["coef"] - coefs["lo"], coefs["hi"] - coefs["coef"]],
            fmt="o", color=NAVY, ecolor=NAVY, capsize=3, markersize=4)
ax.axhline(0, color="#444444", linewidth=0.8)
ax.axvline(2020.5, color=RED, linestyle="--", linewidth=0.9)
ax.text(2020.6, ax.get_ylim()[1], "Cost cap", color=RED, fontsize=8,
        va="top", rotation=90)
ax.set_xlabel("Season (base year 2020)")
ax.set_ylabel(f"Effect of +1 SD pre-cap budget on {OUTCOME}")
ax.set_title("Event study: treatment intensity by season")
save_fig(fig, "fig2_event_study")

# ---------------------------------------------------------------------------
# 4. Efficiency model: change in spending elasticity
# ---------------------------------------------------------------------------
e = cs.dropna(subset=["budget_usd_m", "points"]).copy()
e = e[e["season"] >= 2018]
if DROP_COVID:
    e = e[e["season"] != 2020]
e["ln_points"] = np.log1p(e["points"])           # log(1+points): zeros allowed
e["ln_budget"] = np.log(e["budget_usd_m"])
e["lnb_x_post"] = e["ln_budget"] * e["post2021"]
e = e.set_index(["constructor_id", "season"]).sort_index()
eff_text, eff_rows = "", []
if len(e) > 20:
    eff = fit_fe(e, "ln_points", ["ln_budget", "lnb_x_post"])
    eff_text = str(eff.summary.tables[1])
    for x in ["ln_budget", "lnb_x_post"]:
        eff_rows.append({"term": x, "coef": eff.params[x],
                         "se": eff.std_errors[x], "p": eff.pvalues[x],
                         "n": int(eff.nobs)})
    pd.DataFrame(eff_rows).to_csv(f"{OUT}/table4_efficiency.csv", index=False)
    open(f"{OUT}/table4_efficiency_full.txt", "w").write(eff_text)

    # Dispersion of points per budget unit, pre vs post
    disp = (cs.dropna(subset=["points_per_m"]).groupby("season")["points_per_m"]
              .agg(["mean", "std"]).reset_index())
    disp["cv"] = disp["std"] / disp["mean"]
    disp.to_csv(f"{OUT}/table4b_points_per_m_dispersion.csv", index=False)
else:
    print("Not enough budget data for the efficiency model (need > 20 rows).")

# ---------------------------------------------------------------------------
# 5. Persistence model
# ---------------------------------------------------------------------------
p = cs.dropna(subset=["rank", "rank_lag"]).copy()
p = p[p["season"] >= SAMPLE_START]
if DROP_COVID:
    p = p[p["season"] != 2020]
p["lag_x_post"] = p["rank_lag"] * p["post2021"]
per = smf.ols("rank ~ rank_lag + post2021 + lag_x_post + reset_2009 + reset_2014"
              " + reset_2022", data=p).fit(
    cov_type="cluster", cov_kwds={"groups": p["constructor_id"]})
open(f"{OUT}/table5_persistence.txt", "w").write(str(per.summary().tables[1]))

# ---------------------------------------------------------------------------
# 6. Robustness
# ---------------------------------------------------------------------------
rob = []

# (a) Alternative outcome: qualifying gap to the fastest team (lower = closer)
if ALT_OUTCOME in d.columns:
    dq = d.dropna(subset=[ALT_OUTCOME])
    r = fit_fe(dq, ALT_OUTCOME, ["cap_x_post"])
    rob.append({"check": "Alternative outcome: " + ALT_OUTCOME, "coef": r.params["cap_x_post"],
                "se": r.std_errors["cap_x_post"], "p": r.pvalues["cap_x_post"],
                "n": int(r.nobs)})

# (a2) Alternative outcome: average of both cars' qualifying gap (instead of the fastest car)
if "quali_gap_mean_pct" in d.columns:
    dq2 = d.dropna(subset=["quali_gap_mean_pct"])
    r = fit_fe(dq2, "quali_gap_mean_pct", ["cap_x_post"])
    rob.append({"check": "Alternative outcome: both cars' average qualifying gap",
                "coef": r.params["cap_x_post"], "se": r.std_errors["cap_x_post"],
                "p": r.pvalues["cap_x_post"], "n": int(r.nobs)})

# (b) Excluding 2022 (the reset year)
d_no22 = d[d.index.get_level_values(1) != 2022]
r = fit_fe(d_no22, OUTCOME, ["cap_x_post"])
rob.append({"check": "Exclude 2022", "coef": r.params["cap_x_post"],
            "se": r.std_errors["cap_x_post"], "p": r.pvalues["cap_x_post"],
            "n": int(r.nobs)})

# (c) Excluding 2020 and 2021 (COVID)
d_nocovid = d[~d.index.get_level_values(1).isin([2020, 2021])]
r = fit_fe(d_nocovid, OUTCOME, ["cap_x_post"])
rob.append({"check": "Exclude 2020-2021", "coef": r.params["cap_x_post"],
            "se": r.std_errors["cap_x_post"], "p": r.pvalues["cap_x_post"],
            "n": int(r.nobs)})

# (d) Budget tercile instead of continuous z-score
tmp = d.copy()
terc = tmp.groupby(level=0)["pre_cap_budget_z"].first().rank(pct=True)
tmp["high_budget"] = tmp.index.get_level_values(0).map((terc > 2 / 3).astype(int))
tmp["hb_x_post"] = tmp["high_budget"] * tmp["post2021"]
r = fit_fe(tmp, OUTCOME, ["hb_x_post"])
rob.append({"check": "Top-tercile budget dummy", "coef": r.params["hb_x_post"],
            "se": r.std_errors["hb_x_post"], "p": r.pvalues["hb_x_post"],
            "n": int(r.nobs)})

# (e) Placebo: fake cap in 2017, pre-2021 data only
pl = d[d.index.get_level_values(1) <= 2020].copy()
pl["placebo"] = pl["pre_cap_budget_z"] * (pl.index.get_level_values(1) >= 2017)
r = fit_fe(pl, OUTCOME, ["placebo"])
rob.append({"check": "Placebo cap in 2017 (data to 2020)",
            "coef": r.params["placebo"], "se": r.std_errors["placebo"],
            "p": r.pvalues["placebo"], "n": int(r.nobs)})

# (f) Dose-response: how far above the cap was the team's pre-cap budget? ($100m units)
#     and (g) simple above-cap dummy. The cap only binds for teams spending above it.
if "pre_cap_budget" in d.columns:
    ex = d.copy()
    ex["excess"] = (ex["pre_cap_budget"] - CAP_USD).clip(lower=0) / 100
    ex["excess_x_post"] = ex["excess"] * ex["post2021"]
    ex["above"] = (ex["pre_cap_budget"] > CAP_USD).astype(int)
    ex["above_x_post"] = ex["above"] * ex["post2021"]
    for lab, col in [("Excess over cap ($100m units)", "excess_x_post"),
                     ("Above-cap dummy", "above_x_post")]:
        try:
            r = fit_fe(ex, OUTCOME, [col])
            rob.append({"check": lab, "coef": r.params[col], "se": r.std_errors[col],
                        "p": r.pvalues[col], "n": int(r.nobs)})
        except Exception as err:
            print(f"Skipped '{lab}': {err}")

pd.DataFrame(rob).to_csv(f"{OUT}/table6_robustness.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Leave-one-team-out (does one team drive the DiD result?)
# ---------------------------------------------------------------------------
loo = []
all_teams = d.index.get_level_values(0).unique()
for t in all_teams:
    sub = d[d.index.get_level_values(0) != t]
    try:
        r = fit_fe(sub, OUTCOME, ["cap_x_post"])
        loo.append({"dropped_team": t, "coef": r.params["cap_x_post"],
                    "se": r.std_errors["cap_x_post"], "p": r.pvalues["cap_x_post"],
                    "n": int(r.nobs)})
    except Exception as err:
        print(f"Leave-one-out skipped {t}: {err}")
loo = pd.DataFrame(loo).sort_values("coef")
loo["lo"] = loo["coef"] - 1.96 * loo["se"]
loo["hi"] = loo["coef"] + 1.96 * loo["se"]
loo.to_csv(f"{OUT}/table7_leave_one_team_out.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ypos = np.arange(len(loo))
ax.errorbar(loo["coef"], ypos, xerr=[loo["coef"] - loo["lo"], loo["hi"] - loo["coef"]],
            fmt="o", color=NAVY, ecolor=NAVY, capsize=3, markersize=4)
ax.axvline(0, color="#444444", linewidth=0.8)
ax.axvline(obs, color=RED, linestyle="--", linewidth=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(loo["dropped_team"])
ax.set_xlabel(f"Budget x post-2021 coefficient on {OUTCOME} (dashed = all teams)")
ax.set_title("Leave-one-team-out estimates")
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_leave_one_out.png")
fig.savefig(f"{OUT}/fig3_leave_one_out.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7b. Team-by-team qualifying gap (shows which teams drive the pattern)
# ---------------------------------------------------------------------------
tg = (cs[cs["season"] >= 2014].dropna(subset=["quali_gap_pct"])
        .pivot_table(index="season", columns="constructor_id", values="quali_gap_pct"))
tg.to_csv(f"{OUT}/team_quali_gap_by_season.csv")
fig, ax = plt.subplots(figsize=(8, 5))
for t in tg.columns:
    colour, lw, z = GREY, 1.0, 1
    if t == "mercedes":
        colour, lw, z = RED, 2.0, 3
    elif t == "red_bull":
        colour, lw, z = NAVY, 2.0, 3
    ax.plot(tg.index, tg[t], color=colour, linewidth=lw, zorder=z, alpha=0.9 if z == 3 else 0.7)
    last = tg[t].dropna()
    if len(last):
        ax.text(last.index[-1] + 0.15, last.iloc[-1], t, fontsize=7, color=colour, va="center")
ax.axvline(2020.5, color=GREY, linestyle="--", linewidth=0.9)
ax.text(2020.6, ax.get_ylim()[1], "Cost cap", rotation=90, va="top", fontsize=8, color=GREY)
ax.set_xlabel("Season")
ax.set_ylabel("Qualifying gap to pole (%)")
ax.set_title("Qualifying gap by team (Mercedes red, Red Bull navy)")
ax.set_xlim(tg.index.min(), tg.index.max() + 1.8)
ax.set_xticks(range(int(tg.index.min()), int(tg.index.max()) + 1, 2))
save_fig(fig, "fig4_team_quali_gaps")

# ---------------------------------------------------------------------------
# 8. Season-level qualifying gap using only teams present in EVERY season (2014+)
#    Removes distortion from teams that entered or left (Haas, Manor, Caterham, ...).
# ---------------------------------------------------------------------------
win = cs[cs["season"] >= 2014].dropna(subset=["quali_gap_pct"])
seasons_all = set(win["season"].unique())
teams_full = [t for t, g in win.groupby("constructor_id") if set(g["season"]) == seasons_all]
print("Teams present in every season since 2014:", teams_full)
if len(teams_full) >= 3:
    bal = (win[win["constructor_id"].isin(teams_full)]
           .groupby("season")["quali_gap_pct"].agg(["mean", "median", "std"]).reset_index())
    bal["trend"] = bal["season"] - bal["season"].min()
    bal["post2021"] = (bal["season"] >= 2021).astype(int)
    bal["reset_2022"] = (bal["season"] == 2022).astype(int)
    rows_b = []
    for yv in ["mean", "median", "std"]:
        m_ = smf.ols(f"{yv} ~ trend + post2021 + reset_2022", data=bal).fit(
            cov_type="HAC", cov_kwds={"maxlags": 2})
        rows_b.append({"outcome": yv, "teams": ",".join(teams_full),
                       "post2021_level_shift": m_.params["post2021"], "se": m_.bse["post2021"],
                       "p": m_.pvalues["post2021"], "trend_per_year": m_.params["trend"],
                       "p_trend": m_.pvalues["trend"], "n": int(m_.nobs)})
    pd.DataFrame(rows_b).to_csv(f"{OUT}/table2e_balanced_teams_its.csv", index=False)

    # Does the drop start in 2021 (cap, aero-testing limits) or in 2022 (new aero rules)?
    bal["post2022"] = (bal["season"] >= 2022).astype(int)
    rows_s = []
    for yv in ["mean", "median"]:
        m_ = smf.ols(f"{yv} ~ trend + post2021 + post2022", data=bal).fit(
            cov_type="HAC", cov_kwds={"maxlags": 2})
        rows_s.append({"outcome": yv,
                       "shift_from_2021": m_.params["post2021"], "p_2021": m_.pvalues["post2021"],
                       "extra_shift_from_2022": m_.params["post2022"], "p_2022": m_.pvalues["post2022"],
                       "trend_per_year": m_.params["trend"], "n": int(m_.nobs)})
    pd.DataFrame(rows_s).to_csv(f"{OUT}/table2f_split_2021_vs_2022.csv", index=False)

    # Same level-shift model without 2020 (COVID season) in the pre-period
    rows_c = []
    bal_nc = bal[bal["season"] != 2020]
    for yv in ["mean", "median"]:
        m_ = smf.ols(f"{yv} ~ trend + post2021 + reset_2022", data=bal_nc).fit(
            cov_type="HAC", cov_kwds={"maxlags": 2})
        rows_c.append({"outcome": yv, "post2021_level_shift": m_.params["post2021"],
                       "se": m_.bse["post2021"], "p": m_.pvalues["post2021"],
                       "trend_per_year": m_.params["trend"], "n": int(m_.nobs)})
    pd.DataFrame(rows_c).to_csv(f"{OUT}/table2g_balanced_excluding_2020.csv", index=False)

    # Same level-shift model using the AVERAGE of both cars' gap instead of the fastest car
    if "quali_gap_mean_pct" in cs.columns:
        wm = cs[(cs["season"] >= 2014) & cs["constructor_id"].isin(teams_full)].dropna(
            subset=["quali_gap_mean_pct"])
        bm2 = wm.groupby("season")["quali_gap_mean_pct"].agg(["mean", "median", "std"]).reset_index()
        bm2["trend"] = bm2["season"] - bm2["season"].min()
        bm2["post2021"] = (bm2["season"] >= 2021).astype(int)
        bm2["reset_2022"] = (bm2["season"] == 2022).astype(int)
        rows_m = []
        for yv in ["mean", "median"]:
            m_ = smf.ols(f"{yv} ~ trend + post2021 + reset_2022", data=bm2).fit(
                cov_type="HAC", cov_kwds={"maxlags": 2})
            rows_m.append({"outcome": yv, "measure": "both cars average",
                           "post2021_level_shift": m_.params["post2021"], "se": m_.bse["post2021"],
                           "p": m_.pvalues["post2021"], "trend_per_year": m_.params["trend"],
                           "n": int(m_.nobs)})
        pd.DataFrame(rows_m).to_csv(f"{OUT}/table2h_balanced_both_cars_its.csv", index=False)
    bal.to_csv(f"{OUT}/season_quali_gap_balanced_teams.csv", index=False)

# ---------------------------------------------------------------------------
# Descriptives for Table 1
# ---------------------------------------------------------------------------
desc_cols = [c for c in ["points", "points_share", "rank", "quali_gap_pct",
                         "budget_usd_m", "points_per_m"] if c in cs.columns]
cs[cs["season"] >= DID_START][desc_cols].describe().T.to_csv(
    f"{OUT}/table1_descriptives.csv")

print(f"Done. Tables and figures saved to ./{OUT}")