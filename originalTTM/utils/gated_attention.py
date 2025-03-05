from originalTTM.config.configuration import TTMConfiguration
import torch.nn as nn


class TTMGatedAttention(nn.Module):
  def __init__(self, config: TTMConfiguration, in_size: int, out_size: int):
    """
    Args:
      in_size (`int`): The input size.
      out_size (`int`): The output size.
    """
    super().__init__()
    
    self.attn_layer = nn.Linear(in_size, out_size)
    self.attn_softmax = nn.Softmax(dim=-1)


  def forward(self, inputs):
    """
    Args:
      inputs `torch.Tensor` of shape [B, c, patches, d_model]
    Returns:
      `torch.Tensor` of shape [B, c, patches, d_model]
    """
    attn_weight = self.attn_softmax(self.attn_layer(inputs))
    inputs = inputs * attn_weight
    return inputs