# How Accurate Is Hourly Bike Sharing Demand Estimation?
Linear regression vs kNN for hourly bike-sharing demand.

ESE 5420 final project, group ESE_5420_112 (Jinge, Shuo, Yujie).

Given observed weather and calendar information, how accurately do linear regression and kNN estimate hourly bike rentals, and which one does better during weekday rush hours?

## Reproduce
```bash
conda env create -f environment.yml && conda activate ese5420
python bike.py   # data checks + split sanity
python run.py    # time-ordered CV tuning and validation results -> results/val.md, results/baseline.png
```
`python run.py test` is the final test evaluation: run it once, after the analysis is frozen (see `report.md`, section 5).

## Files
- `bike.py`: data loading, day-based split, preprocessing pipeline, baseline, metrics.
- `run.py`: every experiment and figure; outputs go to `results/`.
- `checkpoint/`: baseline checkpoint submission (`baseline.txt`, `baseline.png`).
- `report.md`: results and analysis so far. `plan.md`: design and milestones.
- `proposal.md`: the submitted proposal; `ESE5420_Topic Proposal.docx`: Word version of the proposal (Oct 16).
- `results/`: current validation outputs; `results/2026-10-16/` and `results/2026-11-08/` hold the earlier test scorings.
- `ckpt/`, `save_ckpt.py`: archived models of the 2026-10-16 setup (k = 5); superseded.

## Data
UCI Bike Sharing Dataset: Fanaee-T, H. (2013), https://doi.org/10.24432/C5W894, licensed CC BY 4.0. The original files are in `data/raw/` (unmodified); `data/README.md` has the download command.
