# from src.architectures.ttm import TTM
from src.architectures.timesfm import TimesFM
# from src.architectures.tcn import TCN
# from src.architectures.static import FineTuneStaticSM

# ttm = TTM()
timesfm = TimesFM()
# tcn = TCN()
# static = FineTuneStaticSM()


# TTM Zero Shot Variant
# zeroshot_trainer = ttm.zeroshot(new=True)

# TTM Few Shot Variant (5% few shot -> ratio=0.05)
# fewshot_trainer = ttm.fewshot(new=True, ratio=0.001, model_name='DNPUs20250225_only_F_01%')

# TimesFM Few Shot Variant
fewshot_trainer = timesfm.fewshot(new=True, ratio=0.001, model_name='DNPUs_20250225')

# TCN
# tcn.train_DNPUs(ratio=0.01, model_name='DNPUs_only_feature_1%_reg')

# Pretrained Static Model 
# static.train_DNPUs(ratio=0.3)

# Light TS

# Informer