from TTMDNPUs.config.configuration import TTMConfiguration
from TTMDNPUs.config.schemas import OutputTTMOutput
import torch
import torch.nn as nn


class TTMOutput(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.flatten = nn.Flatten(start_dim=-2)
    self.dropout_layer = nn.Dropout(config.head_dropout)
    self.channel_projection_block = nn.Linear(config.num_input_channels, config.num_output_channels)
    self.forecast_block = nn.Linear((config.num_patches * config.d_model), config.fl)


  def forward(self, hidden_features):
    """
    Args:
      hidden_features [B, c', patches, d_model] in `common_channel`/`mix_channel` mode.): 
        Input hidden features.
    Returns:
      `torch.Tensor` of shape [B, fl, c'].
    """
    hidden_features = self.flatten(hidden_features)                   # [B, c', patches*d_model]
    hidden_features = self.dropout_layer(hidden_features)             # [B, c', patches*d_model]
    # hidden_features = hidden_features.transpose(-1, -2)               # [B, patches*d_model, c]
    # hidden_features = self.channel_projection_block(hidden_features)  # [B, patches*d_model, c']
    # hidden_features = hidden_features.transpose(-1, -2)               # [B, c', patches*d_model]
    forecast = self.forecast_block(hidden_features)                   # [B, c', fl]
    if isinstance(forecast, tuple):
      forecast = tuple(z.transpose(-1, -2) for z in forecast)
    else:
      forecast = forecast.transpose(-1, -2)               # [B, fl, c']

    return OutputTTMOutput(forecast=forecast)