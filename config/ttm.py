'''
This file is only used by these files.
 - "src/architectures/ttm.py"
 - "src/utils/prepare_data.py".
'''

# Hyper Parameter
TTM_EPOCHS = 100
TTM_LEARNING_RATE = 1.0e-03
TTM_BATCH_SIZE = 256

# Forecasting parameters
TTM_SL = 512                # sl(sequence length), Supported: 512/1024/1536
TTM_FL = 1                  # fl(forcase length),Decide FACF, Supported: ~720
FEWSHOT_FRACTION = 1    # Decide FACFで決定

# Model Structure (Static)
TTM_HIDDEN_LAYERS = [90, 90, 90, 90, 90]
TTM_BATCH_NORM = False
TTM_ACTIVATION_FUNCTION = 'relu'

# TTM (common)
TTM_MODEL_REVISION = "main"
TTM_LOSS = 'mse'  # 'mse' or 'mae'
TTM_PER_DEVICE_EVAL_BATCH_SIZE = 64

# TTM (few-shot variant)
TTM_FEW_DROOPOUT = 0.7
TTM_FEW_LEARNING_RATE = 0.001
TTM_FEW_NUM_EPOCHS = 50 # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
TTM_FEW_BATCH_SIZE = 64