# from src.architectures.torch.ttm import TTM
# from src.architectures.torch.timesfm import TimesFM
from src.architectures.configuration import TCNConfiguration
from src.architectures.torch.tcn import TCN
from src.architectures.lightning.tcn import TCNLightning
# from src.architectures.torch.static import FineTuneStaticSM


# TTM
# ttm = TTM()

# zeroshot_trainer = ttm.zeroshot(new=False)
# ttm.DNPUs(zeroshot_trainer, 'zero')

# fewshot_trainer = ttm.fewshot(new=False, model_name='DNPUs20250225_only_F_01%')
# ttm.DNPUs(fewshot_trainer, 'few')

# TimesFM
# timesfm = TimesFM()
# zeroshot_trainer = timesfm.zeroshot()
# timesfm.DNPUs(zeroshot_trainer, type_='zero')
# timesfm.ettm1(zeroshot_trainer, type_='zero')

# fewshot_trainer = timesfm.fewshot(new=False, ratio=0.001, model_name='DNPUs_20250225')
# timesfm.DNPUs(fewshot_trainer, type_='few', ratio=0.001)
# timesfm.ettm1(fewshot_trainer, type_='few', ratio=0.05)

# TCN
tcn_config = TCNConfiguration()
tcn_config.ratio = 0.01
tcn_config.model_name = 'DNPUs_only_feature_1%_tmp2'
tcn = TCN(tcn_config)
tcn_lightning = TCNLightning(tcn_config)

tcn.test_DNPUs()
# tcn_lightning.test_DNPUs()

# Pretrained Static Model
# static = FineTuneStaticSM()
# static.test_DNPUs(0.3, 'DNPUs.pt')

# LightTS