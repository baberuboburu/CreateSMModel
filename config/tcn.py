'''
This file is only used by these files.
 - "src/architectures/tcn.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameter (DNPUs)
TCN_EPOCHS = 50                     # Max Epoch
TCN_LEARNING_RATE = 1e-4            # Max Learning Rate
TCN_DROPOUT = 0.5
TCN_KERNEL_SIZE = 2

TCN_BATCH_SIZE = 64
TCN_SL = 10
TCN_FL = 1
TCN_NUM_INPUTS = 7                  # Number of the input channels（ex: this value is 1 when the time seriese data with 1 dimension）
TCN_NUM_OUTPUTS = 1                 # Number of the forcast channels（ex: this value is 1 when you want to predict the time series data with 1 dimension）
TCN_NUM_CHANNELS = [8, 16, 32, 64]  # Configuration of the convolutional layers

TCN_MIN_DELTA = 1e-5