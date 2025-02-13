from src.architectures.ttm import TTM
from src.architectures.timesfm import TimesFM
from src.architectures.tcn import TCN

ttm = TTM()
timesfm = TimesFM()
tcn = TCN()

# Zero Shot Variant
# zeroshot_trainer = ttm.zeroshot(new=False)
# ttm.DNPUs(zeroshot_trainer, 'zero')

# Few Shot Variant
# fewshot_trainer = ttm.fewshot(new=False, model_name='DNPUs_20250213')
# ttm.DNPUs(fewshot_trainer, 'few')

# TimesFM Zero Shot Variant
# zeroshot_trainer = timesfm.zeroshot()
# timesfm.DNPUs(zeroshot_trainer, type_='zero')
# timesfm.ettm1(zeroshot_trainer, type_='zero')

# TimesFM Few Shot Variant
# fewshot_trainer = timesfm.fewshot(new=False, ratio=0.05, model_name='DNPUs_20250213')
# timesfm.DNPUs(fewshot_trainer, type_='few', ratio=0.05)
# timesfm.ettm1(fewshot_trainer, type_='few', ratio=0.05)

# TCN
tcn.test_DNPUs(ratio=0.05, model_name='DNPUs_20250213')

# NAMMs

# LightTS