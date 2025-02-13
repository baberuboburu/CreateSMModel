'''
This file is only used by these files.
 - "src/architectures/timesfm.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameters
TIMESFM_EPOCHS = 100
TIMESFM_LEARNING_RATE = 1e-5
TIMESFM_WEIGHT_DECAY = 1e-4
TIMESFM_BATCH_SIZE = 32

TIMESFM_PER_CORE_BATCH_SIZE = 32
TIMESFM_NUM_LAYERS = 50          # Fixed
TIMESFM_SL = 512                 # 
TIMESFM_FL = 128                 # You can decide freely
TIMESFM_INPUT_PATCH_LEN = 32     # Fixed
TIMESFM_OUTPUT_PATCH_LEN = 128   # Fixed
TIMESFM_MODEL_DIM = 1280         # 1280 or 
TIMESFM_HUGGINGFACE_REPO_ID = "google/timesfm-2.0-500m-pytorch"