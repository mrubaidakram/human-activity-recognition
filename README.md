# Human Activity Recognition

> A PyTorch MLP for six-class smartphone activity recognition, using subject-separated validation and the dataset’s official test split.

## Goal

A supervised sensor-classification example. Validation subjects are held out from training subjects; final metrics are computed on the provided official test split.

## Dataset

UCI Human Activity Recognition Using Smartphones: 10,299 windows from 30 participants with 561 engineered sensor features. The official archive is downloaded on first run and excluded from Git.

**Official source:** [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
**Data terms:** UCI lists CC BY 4.0. Cite Reyes-Ortiz et al. (2013) and provide attribution.

Raw dataset files are not committed. First run downloads data into the ignored `data/` directory.

## Run

Python 3.10 or newer is required. Create a virtual environment, install the local package, and train:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m har_torch.train --epochs 40 --data-dir data --output-dir artifacts
```

The first run may download the dataset. Checkpoint and metrics output is written to `artifacts/`.

## Reproducibility and evaluation

- Training, validation, and test responsibilities are documented in the training code.
- Random seeds and key hyperparameters are explicit CLI options or constants.
- Test metrics are reported only after model selection on validation data.
- Results are not published in this README until a reproducible run has produced them.

## Stack

Deep learning · PyTorch · sensor data · human activity recognition

## Scope

This is an independently built portfolio project using public benchmark data. It contains no client records, employer code, or claims that it reproduces private production systems. Dataset attribution and license obligations remain with users of the data.

## Validation status

Python source syntax has been checked. Dependencies have not been installed and training has not been run; runtime behavior and model quality are not yet verified.
