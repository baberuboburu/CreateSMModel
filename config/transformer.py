'''
This file is only used by these files.
 - "src/architectures/tcn.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameter (DNPUs)
TRANSFORMER_EPOCHS = 10                     # Max Epoch
TRANSFORMER_LEARNING_RATE = 1e-4            # Max Learning Rate
TRANSFORMER_DROPOUT = 0.1

TRANSFORMER_BATCH_SIZE = 64
TRANSFORMER_SL = 16
TRANSFORMER_FL = 1
TRANSFORMER_NUM_INPUTS = 7                  # Number of the input channels（ex: this value is 1 when the time seriese data with 1 dimension）
TRANSFORMER_NUM_OUTPUTS = 1                 # Number of the forcast channels（ex: this value is 1 when you want to predict the time series data with 1 dimension）
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_D_MODEL = 64

TRANSFORMER_MIN_DELTA = 1e-5