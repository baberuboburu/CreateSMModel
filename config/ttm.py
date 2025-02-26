'''
This file is only used by these files.
 - "src/architectures/ttm.py"
 - "src/utils/prepare_data.py".
'''

# Forecasting parameters
TTM_SL = 16                           # sl(sequence length), Supported: 512/1024/1536
TTM_FL = 1                            # fl(forcase length),Decide FACF, Supported: ~720
TTM_PATCH_SIZE = 2
TTM_PATCH_STRIDE = 2

# TTM (common between zero-shot and few-shot)
TTM_MODEL_REVISION = "main"
TTM_LOSS = 'mse'                      # 'mse' or 'mae'
TTM_PER_DEVICE_EVAL_BATCH_SIZE = 64

# TTM (few-shot variant)
TTM_FEW_DROOPOUT = 0.7
TTM_FEW_LEARNING_RATE = 1.0e-03
TTM_FEW_NUM_EPOCHS = 50               # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
TTM_FEW_BATCH_SIZE = 64