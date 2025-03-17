# from src.architectures.torch.ttm import TTM
# from src.architectures.torch.ttm_dnpus import TTMDNPUs
# from src.architectures.torch.timesfm import TimesFM
from src.architectures.configuration import TCNConfiguration, TransformerConfiguration, OnlyDecoderConfiguration
# from src.architectures.torch.tcn import TCN
from src.architectures.lightning.transformer import TransformerLightning
from src.architectures.lightning.only_decoder import OnlyDecoderTransformerLightning
# from src.architectures.lightning.tcn import TCNLightning
# from src.architectures.torch.static import FineTuneStaticSM


# TTM
# ttm = TTM()
# ttm_dnpus = TTMDNPUs()
# zeroshot_trainer = ttm.zeroshot(new=True)
# fewshot_trainer = ttm.fewshot(new=True, ratio=0.001, model_name='DNPUs20250225_only_F_01%')
# pretrained_trainer = ttm_dnpus.pretrained_model(new=True, ratio=0.001, model_name='DNPUs20250305_only_F_01%')

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

# # Transformer
# transformer_config = TransformerConfiguration()
# transformer_config.ratio = 0.05
# transformer_config.model_name = f'20250317_{transformer_config.ratio * 100}%'
# transformer_lightning = TransformerLightning(transformer_config)
# transformer_lightning.train_DNPUs()

# Only Decoder Transformer
only_decoder_config = OnlyDecoderConfiguration()
only_decoder = OnlyDecoderTransformerLightning(only_decoder_config)
only_decoder_config.ratio = 0.05
only_decoder.train_DNPUs()