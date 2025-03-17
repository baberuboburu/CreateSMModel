'''
This file is only used by these files.
 - "src/architectures/ttm.py"
 - "src/utils/prepare_data.py".
'''

# Forecasting parameters
TTM_DNPUs_SL = 16                           # sl(sequence length), Supported: 512/1024/1536
TTM_DNPUs_FL = 1                            # fl(forcase length),Decide FACF, Supported: ~720
TTM_DNPUs_PATCH_SIZE = 1
TTM_DNPUs_PATCH_STRIDE = 1
TTM_DNPUs_BATCH_SIZE = 64

# TTM (common between zero-shot and few-shot)
TTM_DNPUs_MODEL_REVISION = "main"
TTM_DNPUs_PER_DEVICE_EVAL_BATCH_SIZE = 64

# TTM (few-shot variant)
TTM_DNPUs_FEW_DROOPOUT = 0.7
TTM_DNPUs_FEW_LEARNING_RATE = 1.0e-03
TTM_DNPUs_FEW_NUM_EPOCHS = 50               # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)

# TTM (pretrained model)
TTM_DNPUs_PRETRAINED_DROOPOUT = 0.4
TTM_DNPUs_PRETRAINED_LEARNING_RATE = 1.0e-03
TTM_DNPUs_PRETRAINED_NUM_EPOCHS = 10
TTM_DNPUs_PRETRAINED_BATCH_SIZE = 64