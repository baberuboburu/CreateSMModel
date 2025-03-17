from TTMDNPUs.config.configuration import TTMConfiguration
from TTMDNPUs.config.schemas import OutputTTMTTMMixer, OutputTTMAdaptivePatchMixer
from TTMDNPUs.components.ts_mixer import TSMixer
import copy
import torch
import torch.nn as nn


class TTMMixer(nn.Module):
  def __init__(self, config: TTMConfiguration, mixer_type: str):
    super().__init__()
    self.adaptive_patching_levels = config.adaptive_patching_levels

    # Encoder
    if self.adaptive_patching_levels > 0:
      num_layers = config.encoder_num_layers
      self.mixers = nn.ModuleList(
        [
          TTMAdaptivePatchMixer(config=config, adapt_patch_level=i, mixer_type=mixer_type)
          for i in reversed(range(config.adaptive_patching_levels))
        ]
      )

    # Encoder / Decoder
    else:
      if mixer_type == 'encoder':
        num_layers = config.encoder_num_layers
      elif mixer_type == 'decoder':
        num_layers = config.decoder_num_layers
      else:
        raise ValueError(f'{mixer_type} num layers is not defined.')
      self.mixers = nn.ModuleList([TSMixer(config=config, mixer_type=mixer_type) for _ in range(num_layers)])
  

  def forward(self, hidden_state):
    embedding = hidden_state

    for mixer in self.mixers:
      embedding = mixer(embedding).hidden

    return OutputTTMTTMMixer(embedding=embedding)




class TTMAdaptivePatchMixer(nn.Module):
  def __init__(self, config: TTMConfiguration, adapt_patch_level: int, mixer_type: str = None):
    super().__init__()
    adaptive_patch_config = copy.deepcopy(config)
    self.adapt_patch_level = adapt_patch_level
    adaptive_patch_factor = 2**adapt_patch_level
    self.adaptive_patch_factor = adaptive_patch_factor

    if config.d_model // self.adaptive_patch_factor <= 4:
      # do not allow reduction beyond d_model less than 4
      print(f'【Warning】Disabling adaptive patching at level %s. Either increase d_model or reduce adaptive_patching_levels % {adapt_patch_level}')
      self.adaptive_patch_factor = 1

    if config.d_model % self.adaptive_patch_factor != 0:
      raise ValueError("d_model should be divisible by 2^i, where i varies from 0 to adaptive_patching_levels.")
    
    adaptive_patch_config.num_patches = adaptive_patch_config.num_patches * self.adaptive_patch_factor
    adaptive_patch_config.d_model = adaptive_patch_config.d_model // self.adaptive_patch_factor

    if mixer_type == 'encoder':
      num_layers = config.encoder_num_layers
    elif mixer_type == 'decoder':
      num_layers = config.decoder_num_layers
    else:
      raise ValueError(f'{mixer_type} num layers is not defined.')

    self.mixers = nn.ModuleList([TSMixer(adaptive_patch_config, mixer_type=mixer_type) for i in range(num_layers)])


  def forward(self, hidden: torch.Tensor):
    """
    Args:
      hidden (`torch.Tensor` of shape [B, c x patches, d_model])
        Input tensor to the layer
    Returns:
      `torch.Tensor`: Transformed tensor.
    """
    hidden = torch.reshape(
      hidden,
      (
        hidden.shape[0],
        hidden.shape[1],
        hidden.shape[2] * self.adaptive_patch_factor,
        hidden.shape[3] // self.adaptive_patch_factor,
      ),
    )

    for mixer in self.mixers:
      hidden = mixer(hidden).hidden

    hidden = torch.reshape(
      hidden,
      (
        hidden.shape[0],
        hidden.shape[1],
        hidden.shape[2] // self.adaptive_patch_factor,
        hidden.shape[3] * self.adaptive_patch_factor,
      ),
    )

    return OutputTTMAdaptivePatchMixer(hidden=hidden)