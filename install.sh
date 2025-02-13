#!/bin/bash
pip install pandas datasets deprecated scikit-learn statsmodels
pip install transformers==4.48.0
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --no-deps
pip install 'accelerate>=0.26.0'
pip uninstall numpy -y
pip install numpy==1.24.3
pip install jax jaxlib
pip install tensorflow