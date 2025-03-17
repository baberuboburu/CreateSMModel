# Create Surrogate Model Using Dynamic DNPUs Data

## Overview
This project focuses on building deep learning models using multiple architectures for time-series prediction tasks. It involves data preparation, training, evaluation, and visualization of results. The project follows a structured directory organization to maintain clarity and scalability.

## Directory Structure
```
project_root/
│── main_train.py             # Entry point for training models
│── main_predict.py           # Entry point for making predictions
│── main_girdsearch.py        # Uncompleted File.
│
├── config/                   # Configuration files
│   ├── config.py             # This file has all config vaiate, and then only this file is imported
│   ├── paths.py              # Centralized file path management, You can select columns for input
│   ├── common.py             # Common settings
│   ├── tcn.py                # TCN-specific settings
│   ├── timesfm.py            # TimesFM-specific settings
│   ├── ttm.py                # TTM-specific settings
│   ├── ttm_dnpus.py          # TTM DNPUs-specific settings
│   ├── transformer.py        # Transformer-specific settings
│   ├── decoder_only.py       # Decoder Only Trasnformer-specific settings
│   ├── ttm_dnpus.py          # TTM DNPUs-specific settings
│   ├── armodel.py            # ARModel Task settings
│   ├── narma.py              # NARMA Task settings
│
├── data/                     # Dataset storage
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── IO.csv                # This file is not uploaded to github.
│
├── models/                   # Trained model storage
│   ├── tcn/
│   ├── timesfm/
│   ├── ttm/
│   │   ├── zeroshots/
│   │   ├── fewshots/
│
├── outputs/                  # Prediction & analysis results
│   ├── predictions/          # Model-generated outputs (.png)
│   │   ├── tcn/              # TCN model
│   │   ├── timesfm/          # TiemsFM model (Using a pre-trained model)
│   │   ├── ttm/              # TTM model
│   │   ├── ttm_dnpus/        # TTM model applyed to DNPUs dataset
│   │   ├── transformer/      # Normal Transformer
│   │   ├── only_decoder/     # Normal Transformer with only decoder
│   │   ├── static/           # Static DNPUs model
│   │
│   ├── analysis/             # Dataset analysis results (.png)
│   │   ├── ETTh1/
│   │   ├── IO/
│
├── src/                      # Core project logic
│   ├── architectures/        # Model architectures
│   │   ├── base.py           # Common model functions
│   │   ├── torch/            # written by normal pytorch
│   │   │   ├── tcn.py        # TCN model
│   │   │   ├── timesfm.py    # TimesFM model (Using a pre-trained model)
│   │   │   ├── ttm.py        # TTM model
│   │   │   ├── ttm_dnpus.py  # TTM model applyed to DNPUs dataset
│   │   │   ├── static        # Static DNPUs model
│   │   ├── lightning/        # written by lightning framework
│   │   │   ├── tcn.py        # TCN model
│   │   │   ├── transformer   # Normal Transformer
│   │   │   ├── only_decoder  # Normal Transformer with only decoder.
│   │
│   ├── utils/                # Utility functions
│   │   ├── early_stop.py
│   │   ├── normalize.py
│   │   ├── plot.py
│   │   ├── process.py
│   │   ├── calculate.py
│   │   ├── prepare_data.py
│   │
│   ├── benchmark/            # Baseline models for comparison
│   │   ├── ARmodel.py
│   │   ├── narma.py
│   │   ├── dnpus.py
│   
├── timesfm/                  # TimesFM library
├── tsfm/                     # TTM library
├── TTMDNPUs/                 # TTM library using only features for training then predicting targets.
│
└── .gitignore
│
└── README.md                 # Project documentation
```

## How To Use
### 1. Clone this repository
`git clone git@github.com:baberuboburu/CreateSMModel.git`
### 2. Activate Conda Environment
`conda activate bspy`
### 3. Install Dependencies
`chmod +x install.sh`  
`./install.sh`
### 4. Set Config Variant
You can edit "config/*.py". And choice the best parameters.
### 5. Prepare IO.dat
You must prepare IO.dat (Dynamic DNPUs data) for below file path.  
`./data/IO.dat`
### 6. Train A New Model
`python main_train.py`
### 7. Predict Use A Model
`python main_predict.py`
