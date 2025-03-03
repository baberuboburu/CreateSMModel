# Hyper Parameter (DNPUs)
INFORMER_EPOCHS = 50                     # Max Epoch
INFORMER_LEARNING_RATE = 1e-4            # Max Learning Rate
INFORMER_DROPOUT = 0.5
INFORMER_KERNEL_SIZE = 2

INFORMER_BATCH_SIZE = 64
INFORMER_SL = 10
INFORMER_FL = 1
INFORMER_NUM_INPUTS = 7                  # Number of the input channels（ex: this value is 1 when the time seriese data with 1 dimension）
INFORMER_NUM_OUTPUTS = 1 