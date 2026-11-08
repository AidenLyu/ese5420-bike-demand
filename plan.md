# Project Plan: Hourly Bike Sharing Demand Estimation

> **Status (2026-11-08):** baseline checkpoint done (`checkpoint/`). k is tuned by time-ordered CV inside train; validation results are in `report.md`. The test set was scored early twice (2026-10-16, 2026-11-08; archived in `results/`). The final test evaluation runs once, after the analysis is frozen. Next: early paper draft (Nov 16), final report and presentation (Dec 11).

**Question:** Given observed weather + calendar info, how accurately do linear regression and kNN estimate hourly rentals (`cnt`), and which one does better during weekday rush hours?

## Data
UCI Bike Sharing `hour.csv`: 17,379 hourly rows, 2011-01-01 to 2012-12-31, 731 days (some hours are missing, since 731×24 = 17,544). CC BY 4.0, see `data/README.md`.

## Design (fixed by the proposal; do not change without the team agreeing)
| Item | Decision |
|---|---|
| Target | `cnt` |
| Excluded | `instant` (ID), `casual` + `registered` (leakage: they sum to `cnt`), `season` (redundant with `mnth`), `atemp` (collinear with `temp`) |
| One-hot | `hr`, `weekday`, `mnth`, `weathersit` |
| Binary passthrough | `yr`, `workingday`, `holiday` |
| Standardized | `temp`, `hum`, `windspeed` (scaler fit on **train only**) |
| Split | chronological, by whole day, 60/20/20 → 438 / 146 / 147 days |
| Baseline | OLS linear regression (as in the submitted proposal). Reference: train mean per (`hr`, `workingday`), fallback = overall train mean (`bike.baseline`) |
| Models | `LinearRegression`; `KNeighborsRegressor` with k ∈ {1, 2, 3, 5, 10, 15, 30, 60, 100} picked by time-ordered 4-fold CV inside train (since 2026-11-08, per instructor feedback and the checkpoint rules; was {5, 15, 30, 60} picked on val) |
| Ablation | calendar-only (drop `weathersit`, `temp`, `hum`, `windspeed`) for both models |
| Metrics | test RMSE + MAE, overall and rush (`workingday==1`, `hr` ∈ 7–9, 16–19) |
| Final fit | refit the chosen configs on train+val, evaluate test **once** after the analysis is frozen (it was already scored early on 2026-10-16 and 2026-11-08; see `report.md` section 5) |

All of this is implemented in `bike.py` (`split_by_day`, `make_pipeline`, `baseline`, `evaluate`).

## Deliverables
1. Model comparison table (reference / OLS baseline / kNN × full / calendar-only × overall / rush).
2. Val RMSE vs k plot.
3. Observed-vs-predicted plots: a scatter, plus a time series for one test week.
4. Error analysis: where each model beats or loses to the baseline.

## Risks to discuss in the report
- **Distribution shift:** the test set is all of late 2012, when demand is higher (growth trend). `yr=1` appears in only part of the training data.
- **LR has no `hr × workingday` interaction**, so it cannot produce the two-peak commute shape. In the first val run it still beat the baseline (rush RMSE 211 vs 265), probably because `yr` absorbs the growth that the baseline ignores. An optional extension is to add the interaction term.
- **Baseline ignores `yr`**, so part of its error comes from the growth trend and not from the hour pattern. Say this when comparing.
- **kNN distance:** about 50 one-hot columns dominate the 3 scaled continuous features. Document this, and optionally compare a down-weighted one-hot variant (val only).
- **Framing:** weather is observed, so this is conditional estimation, not forecasting.

## Milestones & owners
| Due | Milestone | Owner |
|---|---|---|
| Oct 30 | Data checks, EDA, splits, baseline | Jinge |
| Nov 9 | Baseline checkpoint (`checkpoint/baseline.txt`, `baseline.png`) | All (Yujie uploads) |
| Nov 13 | Preprocessing, LR + kNN, val tuning, ablation | Yujie |
| Nov 16 | Early paper draft | All |
| Nov 27 | Freeze analysis → single test evaluation, rush-hour analysis, figures | Shuo |
| Dec 11 | Report + presentation | All |
