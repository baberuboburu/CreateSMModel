'''
This file is only used by these files.
 - "src/architectures/finetune_static_sm.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameter (DNPUs)
STATIC_EPOCHS = 50                     # Max Epoch
STATIC_LEARNING_RATE = 1e-4            # Max Learning Rate
STATIC_DROPOUT = 0.5
STATIC_KERNEL_SIZE = 2

STATIC_BATCH_SIZE = 32
STATIC_SL = 1
STATIC_FL = 1
STATIC_NUM_INPUTS = 7                     # Number of the input channels（ex: this value is 1 when the time seriese data with 1 dimension）
STATIC_NUM_OUTPUTS = 1                    # Number of the forcast channels（ex: this value is 1 when you want to predict the time series data with 1 dimension）
STATIC_NUM_LAYERS = [90, 90, 90, 90, 90]  # Configuration of the MLP layers

STATIC_MIN_DELTA = 1e-6