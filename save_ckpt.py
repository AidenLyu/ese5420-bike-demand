"""ARCHIVE (2026-10-16 setup: k = 5 picked on validation; superseded). Rebuilds those models, refit on train+val,
saves them to ckpt/, and checks they reproduce the 2026-10-16 test RMSE (results/2026-10-16/test.md). It reads the
test set: do not use it to choose anything. Load with joblib.load and call .predict(df) on rows from bike.load()."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import bike

OUT = Path(__file__).parent / "ckpt"
K = 5  # selected on validation for both feature sets (results/val.md)
MODELS = {  # name -> (estimator, calendar_only, interact, test RMSE in report.md); same set as run.py specs()
    "lr_full": (LinearRegression(), False, False, 135.5),
    "knn_full": (KNeighborsRegressor(K), False, False, 120.9),
    "lrint_full": (LinearRegression(), False, True, 106.9),
    "lr_calendar": (LinearRegression(), True, False, 142.4),
    "knn_calendar": (KNeighborsRegressor(K), True, False, 150.2),
    "lrint_calendar": (LinearRegression(), True, True, 115.5),
}

OUT.mkdir(exist_ok=True)
tr, va, te = bike.split_by_day(bike.load())
fit = pd.concat([tr, va])
for name, (est, cal, inter, want) in MODELS.items():
    joblib.dump(bike.make_pipeline(est, cal, inter).fit(fit, fit[bike.TARGET]), OUT / f"{name}.joblib")
    pred = joblib.load(OUT / f"{name}.joblib").predict(te)  # reload check
    rmse = np.sqrt(np.mean((pred - te[bike.TARGET].to_numpy()) ** 2))
    assert round(rmse, 1) == want, (name, rmse)
    print(f"{name}: test RMSE {rmse:.1f} OK, {(OUT / f'{name}.joblib').stat().st_size / 1e6:.1f} MB")
print(f"saved to {OUT} with scikit-learn {sklearn.__version__} (load with the same version)")
