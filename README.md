# Create Surrogate Model Using 

## Overview
This project focuses on building deep learning models using multiple architectures for time-series prediction tasks. It involves data preparation, training, evaluation, and visualization of results. The project follows a structured directory organization to maintain clarity and scalability.

## Directory Structure
```
project_root/
│── main_train.py        # Entry point for training models
│── main_predict.py      # Entry point for making predictions
│
├── config/              # Configuration files
│   ├── config.py        # This file has all config vaiate, and then only this file is imported
│   ├── paths.py         # Centralized file path management
│   ├── common.py        # Common settings
│   ├── tcn.py           # TCN-specific settings
│   ├── timesfm.py       # TimesFM-specific settings
│   ├── ttm.py           # TTM-specific settings
│   ├── armodel.py       # ARModel Task settings
│   ├── narma.py         # NARMA Task settings
│
├── data/                # Dataset storage
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── IO.csv           # This file is not uploaded to github.
│
├── models/              # Trained model storage
│   ├── tcn/
│   ├── timesfm/
│   ├── ttm/
│   │   ├── zeroshots/
│   │   ├── fewshots/
│
├── outputs/             # Prediction & analysis results
│   ├── predictions/     # Model-generated outputs (.png)
│   │   ├── tcn/
│   │   ├── timesfm/
│   │   ├── ttm/
│   │
│   ├── analysis/        # Dataset analysis results (.png)
│   │   ├── ETTh1/
│   │   ├── IO/
│
├── src/                 # Core project logic
│   ├── architectures/   # Model architectures
│   │   ├── base.py      # Common model functions
│   │   ├── tcn.py       # TCN model
│   │   ├── timesfm.py   # TimesFM model
│   │   ├── ttm.py       # TTM model
│   │
│   ├── utils/           # Utility functions
│   │   ├── early_stop.py
│   │   ├── normalize.py
│   │   ├── plot.py
│   │   ├── process.py
│   │   ├── calculate.py
│   │   ├── prepare_data.py
│   │
│   ├── benchmark/       # Baseline models for comparison
│   │   ├── ARmodel.py
│   │   ├── narma.py
│   │   ├── dnpus.py
│   
├── timesfm/             # TimesFM library
├── tsfm/                # TTM library
│
└── .gitignore
│
└── README.md            # Project documentation
```

## How To Use
### 1. Clone this repository
`git clone `
### 2. Activate Conda Environment
`conda activate bspy`
### 3. Install Dependencies
`chmod +x install.sh`
`./install.sh`
### 4. Set Config Variant
You can edit "config/*.py". And choice the best parameters.
### 5. Train A New Model
`python main_train.py`
### 6. Predict Use A Model
`python main_predict.py`