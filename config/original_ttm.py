'''
This file is only used by these files.
 - "src/architectures/ttm.py"
 - "src/utils/prepare_data.py".
'''

# Forecasting parameters
ORIGINAL_TTM_SL = 16                           # sl(sequence length), Supported: 512/1024/1536
ORIGINAL_TTM_FL = 1                            # fl(forcase length),Decide FACF, Supported: ~720
ORIGINAL_TTM_PATCH_SIZE = 1
ORIGINAL_TTM_PATCH_STRIDE = 1
ORIGINAL_TTM_BATCH_SIZE = 64

# TTM (common between zero-shot and few-shot)
ORIGINAL_TTM_MODEL_REVISION = "main"
ORIGINAL_TTM_PER_DEVICE_EVAL_BATCH_SIZE = 64

# TTM (few-shot variant)
ORIGINAL_TTM_FEW_DROOPOUT = 0.7
ORIGINAL_TTM_FEW_LEARNING_RATE = 1.0e-03
ORIGINAL_TTM_FEW_NUM_EPOCHS = 50               # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)

# TTM (pretrained model)
ORIGINAL_TTM_PRETRAINED_DROOPOUT = 0.4
ORIGINAL_TTM_PRETRAINED_LEARNING_RATE = 1.0e-03
ORIGINAL_TTM_PRETRAINED_NUM_EPOCHS = 5
ORIGINAL_TTM_PRETRAINED_BATCH_SIZE = 64