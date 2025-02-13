from src.architectures.ttm import TTM
from src.architectures.timesfm import TimesFM
from src.architectures.tcn import TCN

ttm = TTM()
timesfm = TimesFM()
tcn = TCN()

# TTM Zero Shot Variant
# zeroshot_trainer = ttm.zeroshot(new=True)

# TTM Few Shot Variant (5% few shot -> ratio=0.05)
# fewshot_trainer = ttm.fewshot(new=True, ratio=0.05, model_name='DNPUs20250213')

# TimesFM Few Shot Variant
# fewshot_trainer = timesfm.fewshot(new=True, ratio=0.05, model_name='DNPUs_20250213')

# TCN
tcn.train_DNPUs(ratio=0.001, model_name='DNPUs_20250213')