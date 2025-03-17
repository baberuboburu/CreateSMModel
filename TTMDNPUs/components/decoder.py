from TTMDNPUs.config.configuration import TTMConfiguration
from TTMDNPUs.config.schemas import OutputTTMDecoder
from TTMDNPUs.components.ttm_mixer import TTMMixer
import torch.nn as nn


class TTMDecoder(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()
    self.ttm_mixer = TTMMixer(config=config, mixer_type='decoder')
  

  def forward(self, decoder_input):
    decoder_output = self.ttm_mixer(decoder_input).embedding  # [B, c, patches, d_model]
    
    return OutputTTMDecoder(decoder_output=decoder_output)