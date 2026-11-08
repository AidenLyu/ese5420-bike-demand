# Test stage (setup refit on train+val)

Re-scored 2026-11-08 after instructor feedback (post-hoc): widened k grid, circular block bootstrap, kNN − LR as the main answer. The first scoring (2026-10-16, k grid [5, 15, 30, 60]) is archived in results/2026-10-16/.

Test mean cnt: all hours 248.6, weekday rush 511.7. LR (full) predicts < 0 for 0.2% of test hours. Max prediction: LR 574, kNN 891; max observed 977.

## Test comparison

| model | rmse | mae | bias | rush_rmse | rush_mae | rush_bias |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 144.9 | 101.5 | -74.7 | 234.4 | 203.8 | -162.6 |
| LR (full) | 135.5 | 100.3 | -3.5 | 203.6 | 169.4 | -134.9 |
| kNN k=1 (full) | 128.0 | 84.2 | -58.2 | 206.3 | 166.8 | -138.9 |
| LR+hr×wd (full) | 106.9 | 79.5 | -3.2 | 168.1 | 136.9 | -90.3 |
| LR (calendar) | 142.4 | 106.0 | 2.5 | 210.6 | 174.2 | -129.1 |
| kNN k=2 (calendar) | 143.0 | 98.4 | -78.5 | 236.5 | 203.2 | -170.7 |
| LR+hr×wd (calendar) | 115.5 | 86.4 | 2.5 | 177.9 | 143.0 | -84.8 |

## Paired circular block bootstrap over test days (7-day blocks, 2,000 resamples), Δ RMSE [95% CI]

| Δ test RMSE (< 0: first is better) | overall | rush |
| --- | --- | --- |
| kNN − LR (main answer) | -7.6 [-17.5, +2.3] | +2.7 [-11.2, +17.3] |
| kNN − LR+hr×wd | +21.0 [+8.1, +33.2] | +38.2 [+19.2, +55.8] |
| LR − Baseline | -9.4 [-15.9, -1.3] | -30.8 [-39.7, -20.1] |
| kNN − Baseline | -16.9 [-30.7, -0.8] | -28.0 [-49.6, -3.9] |
| LR full − LR calendar (value of weather) | -6.9 [-10.7, -4.0] | -7.0 [-13.1, -1.4] |
| kNN full − kNN calendar (value of weather) | -15.1 [-27.6, -0.9] | -30.2 [-51.0, -7.8] |
| LR+hr×wd − LR (extension) | -28.6 [-32.8, -23.3] | -35.5 [-43.1, -24.5] |

## Block-length check (same bootstrap, other block lengths; 1 = plain day bootstrap)

| comparison | block (days) | overall | rush |
| --- | --- | --- | --- |
| kNN − LR (main answer) | 1 | -7.6 [-15.0, +0.1] | +2.7 [-11.0, +16.7] |
| kNN − LR (main answer) | 3 | -7.6 [-15.8, +1.2] | +2.7 [-12.0, +18.2] |
| kNN − LR (main answer) | 7 | -7.6 [-17.5, +2.3] | +2.7 [-11.2, +17.3] |
| kNN − LR (main answer) | 14 | -7.6 [-18.6, +4.0] | +2.7 [-14.0, +20.8] |
| kNN − LR (main answer) | 21 | -7.6 [-19.0, +5.3] | +2.7 [-16.2, +25.1] |
| kNN − LR+hr×wd | 1 | +21.0 [+13.8, +29.0] | +38.2 [+22.7, +53.7] |
| kNN − LR+hr×wd | 3 | +21.0 [+11.0, +31.5] | +38.2 [+20.9, +55.0] |
| kNN − LR+hr×wd | 7 | +21.0 [+8.1, +33.2] | +38.2 [+19.2, +55.8] |
| kNN − LR+hr×wd | 14 | +21.0 [+6.7, +34.9] | +38.2 [+17.0, +58.5] |
| kNN − LR+hr×wd | 21 | +21.0 [+6.0, +35.6] | +38.2 [+14.3, +60.9] |

## Sensitivity: fit on train only vs refit on train+val (test RMSE)

| model | rmse, train-only fit | rmse, train+val refit | rush_rmse, train-only fit | rush_rmse, train+val refit |
| --- | --- | --- | --- | --- |
| Baseline | 171.9 | 144.9 | 281.2 | 234.4 |
| LR (full) | 145.7 | 135.5 | 231.8 | 203.6 |
| kNN k=1 (full) | 151.1 | 128.0 | 250.6 | 206.3 |
| LR+hr×wd (full) | 122.0 | 106.9 | 198.1 | 168.1 |

## Test RMSE by hour (table view of test_rmse_by_hour.png)

| workingday | hr | Baseline | LR (full) | kNN k=1 (full) | LR+hr×wd (full) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 60.2 | 36.3 | 54.0 | 55.5 |
| 0 | 1 | 48.1 | 39.0 | 39.1 | 57.2 |
| 0 | 2 | 34.6 | 37.7 | 43.6 | 63.2 |
| 0 | 3 | 18.5 | 55.3 | 52.2 | 73.5 |
| 0 | 4 | 6.7 | 70.6 | 37.6 | 79.3 |
| 0 | 5 | 9.2 | 82.9 | 24.8 | 74.9 |
| 0 | 6 | 14.2 | 120.9 | 49.2 | 75.2 |
| 0 | 7 | 44.5 | 209.0 | 62.9 | 72.3 |
| 0 | 8 | 101.8 | 262.6 | 115.1 | 84.4 |
| 0 | 9 | 122.5 | 81.0 | 109.0 | 71.5 |
| 0 | 10 | 164.9 | 143.6 | 105.1 | 100.0 |
| 0 | 11 | 204.2 | 201.6 | 132.7 | 134.3 |
| 0 | 12 | 242.2 | 246.8 | 192.1 | 175.3 |
| 0 | 13 | 238.7 | 246.3 | 215.1 | 173.2 |
| 0 | 14 | 226.2 | 237.3 | 214.5 | 163.2 |
| 0 | 15 | 221.3 | 227.2 | 219.7 | 164.3 |
| 0 | 16 | 223.2 | 187.1 | 201.2 | 165.1 |
| 0 | 17 | 222.2 | 180.3 | 174.1 | 167.6 |
| 0 | 18 | 182.5 | 174.3 | 138.0 | 132.3 |
| 0 | 19 | 137.1 | 131.1 | 106.3 | 103.9 |
| 0 | 20 | 92.2 | 99.4 | 76.7 | 76.5 |
| 0 | 21 | 76.2 | 84.3 | 62.4 | 71.4 |
| 0 | 22 | 64.0 | 73.1 | 56.5 | 69.6 |
| 0 | 23 | 59.0 | 62.2 | 53.3 | 68.0 |
| 1 | 0 | 34.6 | 83.9 | 20.8 | 68.7 |
| 1 | 1 | 13.3 | 90.6 | 76.6 | 75.3 |
| 1 | 2 | 8.6 | 92.3 | 56.9 | 78.4 |
| 1 | 3 | 3.5 | 84.8 | 47.9 | 76.1 |
| 1 | 4 | 3.9 | 77.6 | 27.7 | 73.8 |
| 1 | 5 | 18.3 | 62.5 | 73.5 | 66.4 |
| 1 | 6 | 62.0 | 26.5 | 80.1 | 37.3 |
| 1 | 7 | 186.3 | 167.4 | 171.6 | 122.2 |
| 1 | 8 | 281.0 | 292.4 | 274.0 | 211.1 |
| 1 | 9 | 125.1 | 73.0 | 122.2 | 63.7 |
| 1 | 10 | 75.0 | 61.1 | 63.0 | 33.8 |
| 1 | 11 | 94.2 | 61.8 | 72.6 | 36.6 |
| 1 | 12 | 126.1 | 59.1 | 86.9 | 57.4 |
| 1 | 13 | 120.7 | 62.2 | 96.5 | 53.0 |
| 1 | 14 | 116.6 | 65.6 | 110.2 | 52.4 |
| 1 | 15 | 138.5 | 62.7 | 113.0 | 69.2 |
| 1 | 16 | 186.9 | 91.7 | 156.1 | 107.2 |
| 1 | 17 | 310.0 | 272.1 | 252.9 | 233.9 |
| 1 | 18 | 288.5 | 255.9 | 256.0 | 220.2 |
| 1 | 19 | 201.4 | 152.9 | 156.8 | 139.1 |
| 1 | 20 | 142.7 | 88.6 | 105.1 | 83.8 |
| 1 | 21 | 98.7 | 56.8 | 85.7 | 58.3 |
| 1 | 22 | 78.0 | 49.8 | 86.1 | 51.1 |
| 1 | 23 | 59.3 | 53.7 | 75.8 | 51.4 |

## Test RMSE by weather category (1 clear, 2 mist, 3 light rain/snow)

| weathersit | LR (full) | LR (calendar) | kNN k=1 (full) | kNN k=2 (calendar) | rows |
| --- | --- | --- | --- | --- | --- |
| 1 | 141.9 | 146.9 | 119.9 | 154.2 | 2168 |
| 2 | 122.5 | 125.1 | 139.7 | 123.0 | 1082 |
| 3 | 132.5 | 171.9 | 142.3 | 120.0 | 238 |

## Mean signed error (pred − observed) by test month

| mnth | Baseline | LR (full) | kNN k=1 (full) | LR+hr×wd (full) |
| --- | --- | --- | --- | --- |
| 8 | -114.6 | -18.1 | -41.4 | -17.6 |
| 9 | -130.4 | -25.3 | -76.4 | -25.7 |
| 10 | -106.4 | -17.4 | -79.8 | -17.1 |
| 11 | -38.7 | 11.1 | -58.2 | 11.6 |
| 12 | 7.0 | 28.8 | -33.5 | 29.3 |

## Test days with the largest LR error (daily RMSE)

| date | weekday | workingday | holiday | worst weathersit | hours | observed total | LR total | kNN total | LR RMSE | kNN RMSE | Baseline RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012-09-29 | Sat | 0 | 0 | 2 | 24 | 8555 | 6640 | 5975 | 211.0 | 183.9 | 238.9 |
| 2012-09-23 | Sun | 0 | 0 | 2 | 24 | 7907 | 6440 | 5965 | 205.2 | 120.0 | 217.5 |
| 2012-09-15 | Sat | 0 | 0 | 2 | 24 | 8714 | 6986 | 7278 | 203.1 | 131.8 | 242.4 |
| 2012-10-27 | Sat | 0 | 0 | 2 | 24 | 7852 | 6466 | 6169 | 200.6 | 147.8 | 209.3 |
| 2012-10-20 | Sat | 0 | 0 | 1 | 24 | 8090 | 6738 | 5493 | 199.9 | 156.4 | 222.7 |
| 2012-09-09 | Sun | 0 | 0 | 2 | 24 | 8227 | 6722 | 6001 | 197.1 | 142.6 | 225.9 |

## Test days with the largest kNN error (daily RMSE)

| date | weekday | workingday | holiday | worst weathersit | hours | observed total | LR total | kNN total | LR RMSE | kNN RMSE | Baseline RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012-08-20 | Mon | 1 | 0 | 2 | 24 | 6530 | 5991 | 5933 | 135.8 | 253.6 | 146.5 |
| 2012-11-01 | Thu | 1 | 0 | 3 | 24 | 5986 | 5280 | 2664 | 109.9 | 209.6 | 101.1 |
| 2012-12-01 | Sat | 0 | 0 | 2 | 24 | 5191 | 4661 | 1783 | 126.2 | 209.4 | 69.8 |
| 2012-10-26 | Fri | 1 | 0 | 2 | 24 | 7444 | 6427 | 4502 | 126.1 | 199.7 | 167.3 |
| 2012-09-27 | Thu | 1 | 0 | 3 | 24 | 7393 | 6692 | 4778 | 166.2 | 196.0 | 191.0 |
| 2012-11-30 | Fri | 1 | 0 | 2 | 24 | 5668 | 5343 | 2640 | 91.4 | 192.1 | 88.0 |

## Post-hoc: neighbors of every scored hour, selected-k kNN (% of all neighbors)

| scored on | kNN features | same hour % | working day % | from 2012 % |
| --- | --- | --- | --- | --- |
| val, fit on train | full | 87.5 | 70.1 | 16.4 |
| val, fit on train | calendar | 100.0 | 69.5 | 12.3 |
| test, fit on train+val | full | 94.1 | 69.2 | 40.4 |
| test, fit on train+val | calendar | 100.0 | 68.7 | 8.6 |
