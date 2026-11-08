"""All experiments behind report.md. Outputs go to results/.

    python run.py        validation stage: data checks, EDA, k tuning, ablation, kNN diagnostics
    python run.py test   + refit the frozen setup on train+val and score the test set (run once)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import bike

OUT = Path(__file__).parent / "results"
Y = bike.TARGET
INK, GRAY, MUTED, HAIR, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"Baseline": MUTED, "LR": "#2a78d6", "kNN": "#eb6834", "LR+hr×wd": "#1baf7a"}  # one color per model, every figure
MAIN = [("Baseline", "—"), ("LR", "full"), ("kNN", "full"), ("LR+hr×wd", "full")]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": HAIR, "grid.linewidth": 0.8,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": GRAY,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.labelcolor": GRAY,
})


def specs(k):
    """The frozen model set: (model, features) -> (estimator, calendar_only, interact). k = selected k per feature set."""
    return {(m, fs): (est, fs == "calendar", m == "LR+hr×wd")
            for fs in ("full", "calendar")
            for m, est in (("LR", LinearRegression()), ("kNN", KNeighborsRegressor(k[fs])), ("LR+hr×wd", LinearRegression()))}


def predict_all(fit, df, k):
    preds = {("Baseline", "—"): bike.baseline(fit, df)}
    for key, (est, cal, inter) in specs(k).items():
        preds[key] = bike.make_pipeline(clone(est), cal, inter).fit(fit, fit[Y]).predict(df)
    return preds


def score(preds, df):
    return pd.DataFrame({key: bike.evaluate(df[Y], p, df) for key, p in preds.items()}).T


def name(key, k):
    m, fs = key
    return (f"kNN k={k[fs]}" if m == "kNN" else m) + ("" if fs == "—" else f" ({fs})")


def named(t, k):
    return t.set_axis(pd.Index([name(key, k) for key in t.index], name="model"))


def md(t, fmt="{:.1f}"):
    t = t.reset_index()
    rows = [[fmt.format(v) if isinstance(v, float) else str(v) for v in r] for r in t.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in [list(map(str, t.columns)), ["---"] * t.shape[1], *rows])


def boot(pa, pb, df, L=7, B=2000, seed=0):
    """RMSE(pa) - RMSE(pb) as (estimate, 2.5%, 97.5%): paired circular block bootstrap over days in date order.
    Blocks of L whole days keep within-day and day-to-day correlation; L=7 (one weekly cycle) was fixed before
    looking, other L are the block-length check. L=1 is the plain day bootstrap used on 2026-10-16.
    Pass pb = observed values to get the interval of RMSE(pa) itself."""
    day, D, y = df.dteday.factorize(sort=True)[0], df.dteday.nunique(), df[Y].to_numpy()
    starts = np.random.default_rng(seed).integers(0, D, (B, -(-D // L)))
    idx = ((starts[:, :, None] + np.arange(L)) % D).reshape(B, -1)[:, :D]
    out = {}
    for sub, m in (("overall", slice(None)), ("rush", bike.rush_mask(df))):
        n = np.bincount(day[m], minlength=D)
        sse = [np.bincount(day[m], (p[m] - y[m]) ** 2, minlength=D) for p in (pa, pb)]
        diff = lambda i: np.sqrt(sse[0][i].sum(-1) / n[i].sum(-1)) - np.sqrt(sse[1][i].sum(-1) / n[i].sum(-1))
        lo, hi = np.percentile(diff(idx), [2.5, 97.5])
        out[sub] = (diff(np.arange(D)), lo, hi)
    return out


def ci_str(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def neighbor_mix(fit, df, k, calendar_only=False):
    """What the k nearest training rows of each df row look like (% of all neighbors)."""
    pipe = bike.make_pipeline(KNeighborsRegressor(k), calendar_only).fit(fit, fit[Y])
    idx = pipe[-1].kneighbors(pipe[:-1].transform(df), return_distance=False)
    return {"same hour %": 100 * (fit.hr.to_numpy()[idx] == df.hr.to_numpy()[:, None]).mean(),
            "working day %": 100 * fit.workingday.to_numpy()[idx].mean(), "from 2012 %": 100 * fit.yr.to_numpy()[idx].mean()}


def shade_rush(ax):
    for a, b in ((7, 9), (16, 19)):
        ax.axvspan(a - 0.5, b + 0.5, color=HAIR, alpha=0.6, lw=0, zorder=0)
        ax.text((a + b) / 2, 0.98, "rush", transform=ax.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=8)


def save(fig, file):
    fig.savefig(OUT / file)
    plt.close(fig)


OUT.mkdir(exist_ok=True)
df = bike.load()
tr, va, te = bike.split_by_day(df)
splits = {"train": tr, "val": va, "test": te}

# ---------------- validation stage: nothing below the exit reads test targets ----------------
checks = pd.DataFrame({n: {
    "first day": f"{s.dteday.min():%Y-%m-%d}", "last day": f"{s.dteday.max():%Y-%m-%d}", "days": s.dteday.nunique(),
    "rows": len(s), "missing hours": 24 * s.dteday.nunique() - len(s), "holidays": s.groupby("dteday").holiday.first().sum(),
    "rush rows": bike.rush_mask(s).sum(), "rain/snow rows (weathersit≥3)": (s.weathersit >= 3).sum(),
} for n, s in splits.items()})
growth = tr[tr.dteday.dt.strftime("%m%d") <= "0313"].groupby("yr")[Y].mean()  # same calendar window, both years in train

prof = tr.groupby(["workingday", "hr"])[Y].describe()
fig, ax = plt.subplots(figsize=(7, 3))
for wd, color, label in ((1, INK, "Working days"), (0, MUTED, "Non-working days")):
    p = prof.loc[wd]
    ax.fill_between(p.index, p["25%"], p["75%"], color=color, alpha=0.1, lw=0)
    ax.plot(p.index, p["mean"], color=color, label=f"{label}: mean, IQR band")
shade_rush(ax)
ax.set(xticks=range(0, 24, 3), xlabel="Hour of day", ylabel="Rentals per hour", title="Training period: hourly rentals")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
save(fig, "eda_hourly_profile.png")

rows = []
for fs in ("full", "calendar"):
    for k in bike.K_GRID:
        pipe = bike.make_pipeline(KNeighborsRegressor(k), fs == "calendar").fit(tr, tr[Y])
        rows += [{"features": fs, "k": k, "split": n, **bike.evaluate(s[Y], pipe.predict(s), s)} for n, s in (("train", tr), ("val", va))]
sweep = pd.DataFrame(rows)

# k is tuned by time-ordered 4-fold CV inside the training days (forward chaining: each fold fits on every earlier
# training day and scores the next 73). No random folds; validation and test are not read here.
days = np.sort(tr.dteday.unique())
H = 73
cv = []
for o in range(len(days) - 4 * H, len(days), H):
    f, s = tr[tr.dteday < days[o]], tr[tr.dteday.isin(days[o:o + H])]
    fold = f"{s.dteday.min():%Y-%m-%d} to {s.dteday.max():%Y-%m-%d}"
    for fs in ("full", "calendar"):
        models = [("LR", None, LinearRegression(), False), ("LR+hr×wd", None, LinearRegression(), True)]
        for m, k, est, inter in models + [(f"kNN k={k}", k, KNeighborsRegressor(k), False) for k in bike.K_GRID]:
            p = bike.make_pipeline(est, fs == "calendar", inter).fit(f, f[Y]).predict(s)
            cv.append({"fold": fold, "features": fs, "model": m, "k": k, **bike.evaluate(s[Y], p, s)})
cv = pd.DataFrame(cv)
cv_tab = {c: cv.pivot_table(index=["features", "model"], columns="fold", values=c, sort=False) for c in ("rmse", "rush_rmse")}
for t in cv_tab.values():
    t["mean"] = t.mean(axis=1)
cv_k = cv[cv.k.notna()].groupby(["features", "k"])[["rmse", "rush_rmse"]].mean().reset_index()
k_sel = {fs: int(g.k[g.rmse.idxmin()]) for fs, g in cv_k.groupby("features")}

val_pred = predict_all(tr, va, k_sel)
val_tab = score(val_pred, va)
yv = va[Y].to_numpy()
rmse_ci = {key: boot(val_pred[key], yv, va) for key in MAIN}
val_ci = pd.DataFrame({**{f"RMSE, {name(key, k_sel)}": rmse_ci[key] for key in MAIN},
                       "kNN − LR": boot(val_pred[("kNN", "full")], val_pred[("LR", "full")], va),
                       "kNN − LR+hr×wd": boot(val_pred[("kNN", "full")], val_pred[("LR+hr×wd", "full")], va)}).T.map(ci_str)

rush_va = va[bike.rush_mask(va)]
diag = pd.DataFrame({k: neighbor_mix(tr, rush_va, k) for k in bike.K_GRID}).T.rename_axis("k")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharex=True, sharey=True)
curves = (("CV error (mean of 4 time-ordered folds in train)", cv_k[cv_k.features == "full"], 1),
          ("validation error (fit on all of train)", sweep[(sweep.features == "full") & (sweep.split == "val")], 0.4))
for ax, col, title in zip(axes, ("rmse", "rush_rmse"), ("All hours", "Weekday rush hours")):
    for label, s, alpha in curves:
        ax.plot(s.k, s[col], "o-", color=COLOR["kNN"], alpha=alpha, ms=5, mec=SURFACE, mew=1.5, label=f"kNN, {label}")
    s = curves[0][1]
    ax.plot(k_sel["full"], s[s.k == k_sel["full"]][col].item(), "o", ms=10, mfc="none", mec=INK, mew=1.2,
            label=f"Selected k={k_sel['full']} (min mean CV RMSE)")
    ax.axhline(val_tab.loc[("LR", "full"), col], color=COLOR["LR"], lw=1.2, label="LR, validation error")
    ax.axhline(val_tab.loc[("Baseline", "—"), col], color=COLOR["Baseline"], lw=1.2, label="Baseline, validation error")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set(title=title, xlabel="k (neighbors, log scale)")
axes[0].set_ylabel("RMSE (rentals per hour)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.17))
save(fig, "val_k_curve.png")

# key figure (baseline.png), the proposal's grouped bar chart: models on x, RMSE with 95% interval, all hours vs rush.
# Validation for now; the final version shows test RMSE.
LABEL = {("Baseline", "—"): "Hour × working-day\nmean (reference)", ("LR", "full"): "Linear regression\n(OLS, baseline)",
         ("kNN", "full"): f"kNN, k={k_sel['full']}\n(chosen by CV)", ("LR+hr×wd", "full"): "OLS + hour ×\nworking day"}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x, w = np.arange(len(MAIN)), 0.36
for j, (sub, alpha) in enumerate((("overall", 0.45), ("rush", 1))):
    est, lo, hi = np.array([rmse_ci[key][sub] for key in MAIN]).T
    xs = x + (j - 0.5) * (w + 0.03)
    ax.bar(xs, est, w, color=[COLOR[key[0]] for key in MAIN], alpha=alpha, zorder=2)
    ax.errorbar(xs, est, [est - lo, hi - est], fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=3)
    for xi, e, h in zip(xs, est, hi):
        ax.annotate(f"{e:.0f}", (xi, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", color=GRAY, fontsize=8)
ax.legend(handles=[matplotlib.patches.Patch(color=MUTED, alpha=a, label=l) for a, l in ((0.45, "All hours"), (1, "Weekday rush hours"))],
          loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
ax.set_xticks(x, [LABEL[key] for key in MAIN])
ax.set(ylabel="Validation RMSE (rentals per hour)", title="Validation RMSE, 95% intervals", ylim=(0, 320))
ax.grid(axis="x", visible=False)
fig.text(0, -0.1, "Whiskers: paired circular block bootstrap over validation days (7-day blocks, 2,000 resamples). Fit on 2011-01-01 to 2012-03-13;\n"
         "validation 2012-03-14 to 2012-08-06; kNN k by time-ordered 4-fold CV inside train. Data: UCI Bike Sharing (Fanaee-T, 2013), CC BY 4.0.",
         color=MUTED, fontsize=7.5)
save(fig, "baseline.png")

wide = sweep.pivot_table(index=["features", "k"], columns="split", values=["rmse", "rush_rmse", "bias", "rush_bias"])
wide.columns = [f"{s} {m}" for m, s in wide.columns]
wide = wide[["train rmse", "val rmse", "train rush_rmse", "val rush_rmse", "val bias", "val rush_bias"]]
val_md = "\n\n".join([
    "# Validation stage (k tuned by time-ordered CV inside train; models fit on train, scored on val)",
    "## Data checks", md(checks),
    f"Training data, Jan 1–Mar 13 window: mean cnt 2011 = {growth[0]:.1f}, 2012 = {growth[1]:.1f} ({growth[1] / growth[0] - 1:+.0%}).",
    "## Time-ordered 4-fold CV inside the training days (each fold fits on all earlier training days, scores the next 73)",
    "### RMSE", md(cv_tab["rmse"]), "### Rush-hour RMSE", md(cv_tab["rush_rmse"]),
    f"Selected k (min mean CV RMSE over the 4 folds, grid {bike.K_GRID}): {k_sel}",
    "## Validation comparison (fit on all training days)", md(named(val_tab, k_sel)),
    "## Validation 95% intervals (paired circular block bootstrap over validation days, 7-day blocks, 2,000 resamples)",
    md(val_ci.rename_axis("validation RMSE or Δ RMSE (< 0: first is better)")),
    "## kNN over the grid, fit on train, scored on train and val (for reference; not used to pick k)", md(wide),
    "## Neighbors of val rush-hour points, full-feature kNN (% of the k neighbors)", md(diag),
])
(OUT / "val.md").write_text(val_md + "\n", encoding="utf-8")
print(val_md)

if sys.argv[1:] != ["test"]:
    print("\nValidation stage done. `python run.py test` scores the frozen setup on test (run once).")
    sys.exit()

# ---------------- test stage: frozen setup, refit on train+val, test scored once ----------------
fit = pd.concat([tr, va])
pred = predict_all(fit, te, k_sel)
test_tab = score(pred, te)
rush = bike.rush_mask(te)

sens = pd.concat({"train-only fit": score(predict_all(tr, te, k_sel), te), "train+val refit": test_tab}, axis=1)
sens = sens.loc[MAIN, [(f, m) for m in ("rmse", "rush_rmse") for f in ("train-only fit", "train+val refit")]]
sens.columns = [f"{m}, {f}" for f, m in sens.columns]

comps = {
    "kNN − LR (main answer)": (("kNN", "full"), ("LR", "full")),
    "kNN − LR+hr×wd": (("kNN", "full"), ("LR+hr×wd", "full")),
    "LR − Baseline": (("LR", "full"), ("Baseline", "—")),
    "kNN − Baseline": (("kNN", "full"), ("Baseline", "—")),
    "LR full − LR calendar (value of weather)": (("LR", "full"), ("LR", "calendar")),
    "kNN full − kNN calendar (value of weather)": (("kNN", "full"), ("kNN", "calendar")),
    "LR+hr×wd − LR (extension)": (("LR+hr×wd", "full"), ("LR", "full")),
}
ci = pd.DataFrame({c: boot(pred[a], pred[b], te) for c, (a, b) in comps.items()}).T.map(ci_str).rename_axis("Δ test RMSE (< 0: first is better)")
blocks = pd.DataFrame({(c, L): boot(pred[comps[c][0]], pred[comps[c][1]], te, L) for c in list(comps)[:2]
                       for L in (1, 3, 7, 14, 21)}).T.map(ci_str).rename_axis(["comparison", "block (days)"])

err = {key: pd.Series(p - te[Y].to_numpy(), index=te.index) for key, p in pred.items()}
by_hour = {key: err[key].pow(2).groupby([te.workingday, te.hr]).mean().pow(0.5) for key in MAIN}
month_bias = pd.DataFrame({name(key, k_sel): err[key].groupby(te.mnth).mean() for key in MAIN})
weather_keys = [("LR", "full"), ("LR", "calendar"), ("kNN", "full"), ("kNN", "calendar")]
weather = pd.DataFrame({name(key, k_sel): err[key].pow(2).groupby(te.weathersit).mean().pow(0.5) for key in weather_keys})
weather["rows"] = te.weathersit.value_counts()

d = te.assign(lr=pred[("LR", "full")], knn=pred[("kNN", "full")], base=pred[("Baseline", "—")])
for c in ("lr", "knn", "base"):
    d[f"se_{c}"] = (d[c] - d[Y]) ** 2
g = d.groupby("dteday")
days = pd.DataFrame({
    "weekday": g.dteday.first().dt.day_name().str[:3], "workingday": g.workingday.first(), "holiday": g.holiday.first(),
    "worst weathersit": g.weathersit.max(), "hours": g.size(), "observed total": g[Y].sum(),
    "LR total": g.lr.sum().round().astype(int), "kNN total": g.knn.sum().round().astype(int),
    "LR RMSE": g.se_lr.mean() ** 0.5, "kNN RMSE": g.se_knn.mean() ** 0.5, "Baseline RMSE": g.se_base.mean() ** 0.5,
})
days.index = days.index.strftime("%Y-%m-%d")
top = {m: days.nlargest(6, f"{m} RMSE").rename_axis("date") for m in ("LR", "kNN")}
mix = pd.DataFrame({(stage, fs): neighbor_mix(f, s, k_sel[fs], fs == "calendar")  # post-hoc: why weather flips for kNN
                    for stage, f, s in (("val, fit on train", tr, va), ("test, fit on train+val", fit, te))
                    for fs in ("full", "calendar")}).T.rename_axis(["scored on", "kNN features"])

daily = df.groupby("dteday")[Y].sum()
fig, ax = plt.subplots(figsize=(8.5, 2.8))
for n, s, alpha in (("train", tr, 0), ("validation", va, 0.35), ("test", te, 0.8)):
    a, b = s.dteday.min(), s.dteday.max() + pd.Timedelta(days=1)
    ax.axvspan(a, b, color=HAIR, alpha=alpha, lw=0, zorder=0)
    ax.text(a + (b - a) / 2, 0.98, n, transform=ax.get_xaxis_transform(), ha="center", va="top", color=GRAY)
ax.plot(daily.index, daily, color=MUTED, lw=0.8, label="Daily total")
ax.plot(daily.index, daily.rolling(28, center=True).mean(), color=INK, label="28-day centered mean")
ax.set(ylabel="Rentals per day", title="Daily rentals, 2011–2012, with the chronological split")
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
save(fig, "daily_series.png")

start = te.dteday[te.dteday.dt.dayofweek == 0].min()  # first full Mon–Sun week of the test period (pre-specified)
wk = (te.dteday >= start) & (te.dteday < start + pd.Timedelta(days=7))
t = te.dteday[wk] + pd.to_timedelta(te.hr[wk], unit="h")
fig, ax = plt.subplots(figsize=(8.5, 3))
ax.plot(t, te[Y][wk], color=INK, label="Observed")
for key in MAIN[:3]:
    ax.plot(t, pred[key][wk.to_numpy()], color=COLOR[key[0]], lw=1.4, label=name(key, k_sel))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
ax.set(ylabel="Rentals per hour", title=f"First full test week ({start:%b %d}–{start + pd.Timedelta(days=6):%b %d, %Y}): observed vs predicted")
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
save(fig, "test_week.png")

fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), sharex=True, sharey=True)
lim = (min(-50, min(p.min() for p in pred.values())), max(te[Y].max(), max(p.max() for p in pred.values())) + 20)
for ax, key in zip(axes, MAIN[:3]):
    ax.plot(lim, lim, color=AXIS, lw=1, zorder=1)
    ax.scatter(te[Y][~rush], pred[key][~rush], s=4, color=AXIS, alpha=0.35, lw=0, label="Other hours")
    ax.scatter(te[Y][rush], pred[key][rush], s=6, color=GRAY if key[0] == "Baseline" else COLOR[key[0]], alpha=0.5, lw=0,
               label="Weekday rush hours")
    r = test_tab.loc[key]
    ax.set(xlim=lim, ylim=lim, xlabel="Observed rentals per hour", title=f"{name(key, k_sel)}\nRMSE {r.rmse:.0f} · rush RMSE {r.rush_rmse:.0f}")
    ax.legend(loc="upper left", markerscale=2, handletextpad=0.2)
axes[0].set_ylabel("Predicted rentals per hour")
save(fig, "test_obs_vs_pred.png")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
for ax, wd, title in zip(axes, (1, 0), ("Working days", "Non-working days")):
    for key in MAIN:
        s = by_hour[key].loc[wd]
        ax.plot(s.index, s.values, color=COLOR[key[0]], lw=1.6, label=name(key, k_sel))
    if wd:
        shade_rush(ax)
    ax.set(title=f"Test RMSE by hour: {title.lower()}", xlabel="Hour of day", xticks=range(0, 24, 3))
axes[0].set_ylabel("RMSE (rentals per hour)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
save(fig, "test_rmse_by_hour.png")

test_md = "\n\n".join([
    "# Test stage (setup refit on train+val; kNN k from time-ordered CV inside train)",
    "Earlier test scorings are archived in results/2026-10-16/ (k grid [5, 15, 30, 60], k picked on validation) and "
    "results/2026-11-08/ (widened grid, k picked on validation); report this run as post-hoc relative to both.",
    f"Test mean cnt: all hours {te[Y].mean():.1f}, weekday rush {te[Y][rush].mean():.1f}. "
    f"LR (full) predicts < 0 for {(pred[('LR', 'full')] < 0).mean():.1%} of test hours. "
    f"Max prediction: LR {pred[('LR', 'full')].max():.0f}, kNN {pred[('kNN', 'full')].max():.0f}; max observed {te[Y].max()}.",
    "## Test comparison", md(named(test_tab, k_sel)),
    "## Paired circular block bootstrap over test days (7-day blocks, 2,000 resamples), Δ RMSE [95% CI]", md(ci),
    "## Block-length check (same bootstrap, other block lengths; 1 = plain day bootstrap)", md(blocks),
    "## Sensitivity: fit on train only vs refit on train+val (test RMSE)", md(named(sens, k_sel)),
    "## Test RMSE by hour (table view of test_rmse_by_hour.png)",
    md(pd.DataFrame({name(key, k_sel): by_hour[key] for key in MAIN})),
    "## Test RMSE by weather category (1 clear, 2 mist, 3 light rain/snow)", md(weather),
    "## Mean signed error (pred − observed) by test month", md(month_bias),
    "## Test days with the largest LR error (daily RMSE)", md(top["LR"]),
    "## Test days with the largest kNN error (daily RMSE)", md(top["kNN"]),
    "## Post-hoc: neighbors of every scored hour, selected-k kNN (% of all neighbors)", md(mix),
])
(OUT / "test.md").write_text(test_md + "\n", encoding="utf-8")
print("\n" + test_md)
