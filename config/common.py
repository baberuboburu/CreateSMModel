'''
This file is only used by these files.
 - "src/architectures/ttm.py"
 - "src/architectures/timesfm.py"
 - "src/architectures/tcn.py"
 - "src/utils/prepare_data.py"
 - src/utils/calculate.py
'''

import torch

# Common
BACKEND = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
LOSS = 'mse'