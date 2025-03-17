'''
This file is only used by these files.
 - "src/architectures/tcn.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameter (DNPUs)
ONLY_DECODER_EPOCHS = 5                     # Max Epoch
ONLY_DECODER_LEARNING_RATE = 1e-4            # Max Learning Rate
ONLY_DECODER_DROPOUT = 0.1

ONLY_DECODER_BATCH_SIZE = 64
ONLY_DECODER_SL = 16
ONLY_DECODER_FL = 1
ONLY_DECODER_NUM_INPUTS = 8                  # Number of the input channels（ex: this value is 1 when the time seriese data with 1 dimension）
ONLY_DECODER_NUM_OUTPUTS = 1                 # Number of the forcast channels（ex: this value is 1 when you want to predict the time series data with 1 dimension）
ONLY_DECODER_NUM_LAYERS = 2
ONLY_DECODER_NUM_HEADS = 4
ONLY_DECODER_D_MODEL = 64

ONLY_DECODER_MIN_DELTA = 1e-5