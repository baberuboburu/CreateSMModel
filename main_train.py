# from src.architectures.torch.ttm import TTM
from src.architectures.torch.original_ttm import OriginalTTM
# from src.architectures.torch.timesfm import TimesFM
# from src.architectures.configuration import TCNConfiguration
# from src.architectures.torch.tcn import TCN
# from src.architectures.lightning.tcn import TCNLightning
# from src.architectures.torch.static import FineTuneStaticSM


# TTM
# ttm = TTM()
original_ttm = OriginalTTM()
# zeroshot_trainer = ttm.zeroshot(new=True)
# fewshot_trainer = ttm.fewshot(new=True, ratio=0.001, model_name='DNPUs20250225_only_F_01%')
pretrained_trainer = original_ttm.pretrained_model(new=True, ratio=0.001, model_name='DNPUs20250305_only_F_01%')

# TimesFM Few Shot Variant
# timesfm = TimesFM()
# fewshot_trainer = timesfm.fewshot(new=True, ratio=0.001, model_name='DNPUs_20250225')

# TCN
# tcn_config = TCNConfiguration()
# tcn_config.ratio = 0.01
# tcn_config.model_name = 'DNPUs_only_feature_1%_tmp2'
# tcn = TCN(tcn_config)
# tcn_lightning = TCNLightning(tcn_config)

# tcn.train_DNPUs()
# tcn_lightning.train_DNPUs()

# Pretrained Static Model 
# static = FineTuneStaticSM()
# static.train_DNPUs(ratio=0.3)

# Light TS

# Informer