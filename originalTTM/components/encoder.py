from originalTTM.config.configuration import TTMConfiguration
from originalTTM.config.schemas import OutputTTMEncoder
from originalTTM.components.ttm_mixer import TTMMixer
import math
import torch
import torch.nn as nn
# from transformers.modeling_utils import PreTrainedModel


class TTMEncoder(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.d_model = config.d_model
    self.patcher = nn.Linear(config.patch_length, config.d_model)
    self.ttm_mixer = TTMMixer(config=config, mixer_type='encoder')
    if config.use_positional_encoding:
      self.positional_encoder = TTMPositinalEnxoder(config=config)
  

  def forward(self,past_values: torch.Tensor):

    # flatten [B x patches x d_model]. common_channel/mix_channel: [B x c x patches x d_model]
    patches = self.patcher(past_values)
    if hasattr(self, 'positional_encoder'):
      patches = self.positional_encoder(patches)
    encoder_output = self.ttm_mixer(patches).embedding

    return OutputTTMEncoder(encoder_output=encoder_output)




class TTMPositinalEnxoder(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()
    if config.use_positional_encoding:
      self.position_enc = self._init_pe(config)
    else:
      self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))


  def _init_pe(config: TTMConfiguration):
    # Positional encoding
    if config.positional_encoding_type == "random":
      position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
    elif config.positional_encoding_type == "sincos":
      position_enc = torch.zeros(config.num_patches, config.d_model)
      position = torch.arange(0, config.num_patches).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
      position_enc[:, 0::2] = torch.sin(position * div_term)
      position_enc[:, 1::2] = torch.cos(position * div_term)
      position_enc = position_enc - position_enc.mean()
      position_enc = position_enc / (position_enc.std() * 10)
      position_enc = nn.Parameter(position_enc, requires_grad=False)
    else:
      raise ValueError(
        f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
      )
    return position_enc


  def forward(self, patch_input: torch.Tensor):
    hidden_state = patch_input + self.position_enc  # [B, c, patches, d_model]
    return hidden_state