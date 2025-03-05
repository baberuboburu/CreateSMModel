from originalTTM.config.configuration import TTMConfiguration
from originalTTM.config.schemas import OutputTTMDecoder
from originalTTM.components.ttm_mixer import TTMMixer
import torch.nn as nn


class TTMDecoder(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()
    self.ttm_mixer = TTMMixer(config=config, mixer_type='decoder')
  

  def forward(self, decoder_input):
    decoder_output = self.ttm_mixer(decoder_input).embedding  # [B, c, patches, d_model]
    
    return OutputTTMDecoder(decoder_output=decoder_output)