from originalTTM.config.configuration import TTMConfiguration
from originalTTM.config.schemas import OutputTSMixer, OutputTSPatchMixer, OutputTSFeatureMixer, OutputTSChannelMixer
from originalTTM.utils.mlp import TTMMLP
from originalTTM.utils.gated_attention import TTMGatedAttention
from originalTTM.utils.normalize import TTMNormalize
import torch
import torch.nn as nn


class TSMixer(nn.Module):
  def __init__(self, config: TTMConfiguration, mixer_type: str):
    super().__init__()

    self.num_patches = config.num_patches
    self.mixer_type = mixer_type

    if mixer_type == 'encoder':
      self.mode = config.encoder_mode
    elif mixer_type == 'decoder':
      self.mode = config.decoder_mode
    else:
      raise ValueError(f'mixer type is not in [encoder, decoder]. at class TSMixer()')
    
    if config.num_patches > 1:
      self.patch_mixer = TSPatchMixer(config=config)

    self.feature_mixer = TSFeatureMixer(config=config, mixer_type=mixer_type)

    if self.mode == "mix_channel":
      self.channel_feature_mixer = TSChannelMixer(config=config, mixer_type=mixer_type)


  def forward(self, hidden: torch.Tensor):
    """
    Args:
      hidden (`torch.Tensor` of shape [B, c, patches, d_model])
        Input tensor to the layer.
    Returns:
      `torch.Tensor`: Transformed tensor.
    """
    if self.num_patches > 1:
      hidden = self.patch_mixer(hidden).patch_mixed_hidden

    hidden = self.feature_mixer(hidden).feature_mixed_hidden     # [B, c, patches, d_model]

    if self.mode == "mix_channel":
      hidden = self.channel_feature_mixer(hidden).channel_mixed_hidden

    return OutputTSMixer(hidden=hidden)




class TSPatchMixer(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.norm = TTMNormalize(config)

    self.mlp = TTMMLP(
      config=config,
      in_features=config.num_patches,
      out_features=config.num_patches,
    )

    self.gating_block = TTMGatedAttention(
      config=config,
      in_size=config.num_patches, 
      out_size=config.num_patches
    )


  def forward(self, hidden):
    """
    Args:
      hidden (`torch.Tensor`): Input tensor.
    Returns:
      `torch.Tensor`: Transformed tensor.
    """
    residual = hidden
    hidden = self.norm(hidden)

    # Transpose so that num_patches is the last dimension
    hidden = hidden.transpose(2, 3)
    # print(f'Patch Mixer: {hidden.shape}')
    hidden = self.mlp(hidden)
    hidden = self.gating_block(hidden)
    hidden = hidden.transpose(2, 3)

    patch_mixed_hidden = hidden + residual
    return OutputTSPatchMixer(patch_mixed_hidden=patch_mixed_hidden)





class TSFeatureMixer(nn.Module):
  def __init__(self, config: TTMConfiguration, mixer_type: str):
    super().__init__()

    self.norm = TTMNormalize(config)

    if mixer_type == 'encoder':
      self.mlp = TTMMLP(
        config=config,
        in_features=config.num_input_channels,
        out_features=config.num_input_channels,
      )
      self.gating_block = TTMGatedAttention(
        config=config,
        in_size=config.num_input_channels, 
        out_size=config.num_input_channels
      )
    elif mixer_type == 'decoder':
      self.mlp = TTMMLP(
        config=config,
        in_features=config.num_input_channels,
        out_features=config.num_input_channels,
      )
      self.gating_block = TTMGatedAttention(
        config=config,
        in_size=config.num_input_channels, 
        out_size=config.num_input_channels
      )


  def forward(self, hidden: torch.Tensor):
    """
    Args:
      hidden (`torch.Tensor` of shape [B, c, patches, d_model]):
        input to the MLP layer
    Returns:
      `torch.Tensor` of the same shape as `hidden`
    """
    residual = hidden
    hidden = self.norm(hidden)
    
    hidden = hidden.permute(0, 3, 2, 1)
    # print(f'Feature Mixer: {hidden.shape}')
    hidden = self.mlp(hidden)
    hidden = self.gating_block(hidden)
    hidden = hidden.permute(0, 3, 2, 1)

    feature_mixed_hidden = hidden + residual
    return OutputTSFeatureMixer(feature_mixed_hidden=feature_mixed_hidden)




class TSChannelMixer(nn.Module):
  def __init__(self, config: TTMConfiguration, mixer_type: str):
    super().__init__()

    self.norm = TTMNormalize(config)

    if mixer_type == 'encoder':
      self.mlp = TTMMLP(
        config=config,
        in_features=config.num_input_channels,
        out_features=config.num_input_channels
      )
      self.gating_block = TTMGatedAttention(
        config=config,
        in_size=config.num_input_channels, 
        out_size=config.num_input_channels
      )

    elif mixer_type == 'decoder':
      self.mlp = TTMMLP(
        config=config,
        in_features=config.num_input_channels,
        out_features=config.num_input_channels,
      )
      self.gating_block = TTMGatedAttention(
        config=config,
        in_size=config.num_input_channels, 
        out_size=config.num_input_channels
      )


  def forward(self, hidden: torch.Tensor):
    """
    Args:
      hidden (`torch.Tensor` of shape [B, c, patches, d_model]):
        input to the MLP layer
    Returns:
      `torch.Tensor` of the same shape as `hidden`
    """
    residual = hidden
    hidden = self.norm(hidden)
        
    hidden = hidden.permute(0, 3, 2, 1)
    # print(f'Channel Mixer: {hidden.shape}')
    hidden = self.mlp(hidden)
    hidden = self.gating_block(hidden)
    hidden = hidden.permute(0, 3, 2, 1)

    channel_mixed_hidden = hidden + residual
    return OutputTSChannelMixer(channel_mixed_hidden=channel_mixed_hidden)