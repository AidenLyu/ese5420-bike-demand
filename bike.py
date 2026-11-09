"""Shared data/model utilities for the ESE5420 bike-demand project. Run directly for a self-check."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data/raw/hour.csv"
TARGET = "cnt"
CATEGORICAL = ["hr", "weekday", "mnth", "weathersit"]
BINARY = ["yr", "workingday", "holiday"]
CONTINUOUS = ["temp", "hum", "windspeed"]
WEATHER = ["weathersit", "temp", "hum", "windspeed"]
K_GRID = [1, 2, 3, 5, 10, 15, 30, 60, 100]  # widened 2026-11-08 per instructor feedback (test was first scored with [5, 15, 30, 60])
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]  # 07:00-09:59 and 16:00-19:59


def load():
    df = pd.read_csv(DATA, parse_dates=["dteday"])
    return df.assign(hr_wd=df.hr + 24 * df.workingday)  # hr x workingday cell, for the LR interaction extension


def split_by_day(df, frac=(0.6, 0.2)):
    """Chronological split on whole days -> (train, val, test)."""
    days = np.sort(df.dteday.unique())
    a, b = int(len(days) * frac[0]), int(len(days) * sum(frac))
    return (df[df.dteday.isin(days[:a])], df[df.dteday.isin(days[a:b])], df[df.dteday.isin(days[b:])])


def features(calendar_only=False, interact=False):
    """(one-hot, binary, standardized) columns; interact swaps hr for the hr x workingday cell."""
    drop = set(WEATHER) if calendar_only else set()
    cat = ["hr_wd" if c == "hr" and interact else c for c in CATEGORICAL if c not in drop]
    return cat, BINARY, [c for c in CONTINUOUS if c not in drop]


def make_pipeline(model, calendar_only=False, interact=False):
    cat, binary, cont = features(calendar_only, interact)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),  # dense -> exact lstsq for LR
        ("bin", "passthrough", binary),
        ("num", StandardScaler(), cont),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def baseline(train, df):
    """Train mean per (hr, workingday); falls back to overall train mean."""
    means = train.groupby(["hr", "workingday"])[TARGET].mean().rename("pred")
    pred = df.join(means, on=["hr", "workingday"])["pred"]
    return pred.fillna(train[TARGET].mean()).to_numpy()


def rush_mask(df):
    return ((df.workingday == 1) & df.hr.isin(RUSH_HOURS)).to_numpy()


def evaluate(y, pred, df):
    """RMSE, MAE and bias (mean of pred - y), overall and on weekday rush hours."""
    e, out = np.asarray(pred) - np.asarray(y), {}
    for prefix, m in [("", slice(None)), ("rush_", rush_mask(df))]:
        out |= {prefix + "rmse": np.sqrt(np.mean(e[m] ** 2)), prefix + "mae": np.mean(np.abs(e[m])), prefix + "bias": np.mean(e[m])}
    return out


if __name__ == "__main__":
    df = load()
    assert len(df) == 17379 and not df.isna().any().any() and not df.duplicated(["dteday", "hr"]).any()
    assert (df.casual + df.registered == df.cnt).all()
    d = df.dteday.dt  # calendar columns agree with the date; workingday = Mon-Fri and not a holiday
    assert ((d.dayofweek + 1) % 7 == df.weekday).all() and (d.year - 2011 == df.yr).all() and (d.month == df.mnth).all()
    assert (df.workingday == (df.weekday.between(1, 5) & (df.holiday == 0))).all()
    used = {c for flag in (False, True) for cols in features(interact=flag) for c in cols}
    assert not used & {"instant", "casual", "registered", TARGET}
    tr, va, te = split_by_day(df)
    assert tr.dteday.max() < va.dteday.min() and va.dteday.max() < te.dteday.min()
    assert len(tr) + len(va) + len(te) == len(df)
    n = [s.dteday.nunique() for s in (tr, va, te)]
    assert abs(n[0] / sum(n) - 0.6) < 0.01, n
    assert rush_mask(va).sum() > 0
    from sklearn.linear_model import LinearRegression  # scaler statistics come from the fitting data only
    assert np.allclose(make_pipeline(LinearRegression()).fit(tr, tr[TARGET])["pre"].named_transformers_["num"].mean_, tr[CONTINUOUS].mean())
    print("days train/val/test:", n, "| ranges:", [f"{s.dteday.min():%Y-%m-%d}..{s.dteday.max():%Y-%m-%d}" for s in (tr, va, te)])
    print("baseline val:", {k: round(v, 1) for k, v in evaluate(va[TARGET], baseline(tr, va), va).items()})
