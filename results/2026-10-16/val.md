# Validation stage (models fit on train, scored on val)

## Data checks

| index | train | val | test |
| --- | --- | --- | --- |
| first day | 2011-01-01 | 2012-03-14 | 2012-08-07 |
| last day | 2012-03-13 | 2012-08-06 | 2012-12-31 |
| days | 438 | 146 | 147 |
| rows | 10389 | 3502 | 3488 |
| missing hours | 123 | 2 | 40 |
| holidays | 13 | 3 | 5 |
| rush rows | 2085 | 707 | 690 |
| rain/snow rows (weathersit≥3) | 934 | 250 | 238 |

Training data, Jan 1–Mar 13 window: mean cnt 2011 = 66.8, 2012 = 147.9 (+121%).

## kNN k sweep (selection uses the grid {5, 15, 30, 60} only; the other k are exploratory)

| features | k | train rmse | val rmse | train rush_rmse | val rush_rmse | val bias | val rush_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calendar | 5 | 53.6 | 149.5 | 80.6 | 238.9 | -101.5 | -205.0 |
| calendar | 15 | 86.0 | 159.0 | 133.3 | 240.8 | -111.5 | -208.0 |
| calendar | 30 | 81.5 | 185.2 | 127.9 | 296.1 | -117.6 | -263.2 |
| calendar | 60 | 92.1 | 201.5 | 144.4 | 332.5 | -107.3 | -296.6 |
| full | 1 | 0.5 | 149.4 | 1.1 | 236.5 | -84.8 | -190.2 |
| full | 2 | 44.6 | 149.5 | 67.9 | 241.4 | -82.8 | -202.5 |
| full | 3 | 54.7 | 153.0 | 84.4 | 250.9 | -83.3 | -215.4 |
| full | 5 | 65.3 | 158.3 | 103.4 | 267.4 | -82.1 | -232.9 |
| full | 7 | 69.9 | 161.3 | 112.0 | 274.1 | -81.2 | -240.2 |
| full | 10 | 74.1 | 164.2 | 119.5 | 281.5 | -81.3 | -247.7 |
| full | 15 | 77.5 | 167.4 | 124.3 | 289.4 | -81.3 | -256.3 |
| full | 20 | 79.2 | 170.2 | 126.3 | 295.2 | -81.1 | -261.7 |
| full | 30 | 81.9 | 173.5 | 130.5 | 302.5 | -80.5 | -269.2 |
| full | 45 | 85.1 | 177.6 | 135.6 | 311.1 | -80.4 | -276.9 |
| full | 60 | 88.0 | 180.9 | 140.5 | 318.1 | -80.5 | -283.3 |
| full | 90 | 92.3 | 185.2 | 147.5 | 327.3 | -80.2 | -291.5 |
| full | 120 | 95.1 | 187.9 | 151.8 | 332.3 | -80.1 | -296.0 |

Selected k (min val RMSE on the grid): {'calendar': 5, 'full': 5}

## Validation comparison

| model | rmse | mae | bias | rush_rmse | rush_mae | rush_bias |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 169.9 | 126.0 | -120.6 | 264.9 | 240.1 | -232.1 |
| LR (full) | 139.2 | 106.2 | -18.2 | 210.9 | 178.0 | -166.3 |
| kNN k=5 (full) | 158.3 | 114.2 | -82.1 | 267.4 | 237.3 | -232.9 |
| LR+hr×wd (full) | 114.1 | 88.4 | -17.8 | 173.4 | 144.2 | -128.2 |
| LR (calendar) | 145.6 | 112.8 | -13.4 | 214.4 | 180.6 | -163.2 |
| kNN k=5 (calendar) | 149.5 | 109.3 | -101.5 | 238.9 | 214.7 | -205.0 |
| LR+hr×wd (calendar) | 121.7 | 95.5 | -13.2 | 178.7 | 148.2 | -125.0 |

## Neighbors of val rush-hour points, full-feature kNN (% of the k neighbors)

| k | same hour % | working day % | from 2012 % |
| --- | --- | --- | --- |
| 5 | 62.8 | 99.5 | 17.5 |
| 15 | 47.8 | 99.0 | 15.9 |
| 30 | 40.8 | 98.2 | 17.9 |
| 60 | 34.4 | 97.0 | 18.7 |
