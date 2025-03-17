from TTMDNPUs.config.configuration import TTMConfiguration
import torch
import torch.nn as nn


class TTMMLP(nn.Module):
  def __init__(self, config: TTMConfiguration, in_features: int, out_features: int):
    super().__init__()

    num_hidden = in_features * config.mlp_expansion
    self.fc1 = nn.Linear(in_features, num_hidden)
    self.dropout1 = nn.Dropout(config.dropout)
    self.fc2 = nn.Linear(num_hidden, out_features)
    self.dropout2 = nn.Dropout(config.dropout)


  def forward(self, inputs: torch.Tensor):
    """
    Args:
      inputs (`torch.Tensor` of shape [B, c, patches, d_model])):
        Input to the MLP layer.
    Returns:
      `torch.Tensor` of the same shape as `inputs`
    """
    # print(inputs.shape)          # torch.Size([64, 1, 48, 8])
    # print(self.fc1.in_features)  # Expect to torch.Size([64, 1, 48, 32])
    inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
    inputs = self.fc2(inputs)
    mlp_output = self.dropout2(inputs)
    return mlp_output