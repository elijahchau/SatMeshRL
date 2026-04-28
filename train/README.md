# Q-learning Training

This folder contains scripts for training Q-learning on real-world snapshots
built from the Starlink TLE dataset.

## What it does

- Loads the first N satellites from `data/starlink_tle.txt`
- Builds snapshots at 1, 5, and 10 minutes
- Samples Poisson queue delays and applies link costs
- Trains Q-learning for multiple episode counts
- Saves Q-tables, cost curves, and training stats

## Run

```powershell
python train\train_qlearning_snapshots.py
```

Outputs are written to `train\outputs`.
