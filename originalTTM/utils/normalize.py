from originalTTM.config.configuration import TTMConfiguration
import torch
import torch.nn as nn


class TTMNormalize(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.norm_type = config.norm_type

    if "batch" in config.norm_type.lower():
      self.norm = TTMBatchNorm(config)
    else:
      self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
  

  def forward(self, inputs: torch.Tensor):
    """
    Args:
      inputs (`torch.Tensor` of shape [B, c, patches, d_model]):
        Input to the normalization layer.
    Returns:
      `torch.Tensor` of shape [B, c, patches, d_model]
    """
    if "batch" in self.norm_type.lower():
      # inputs_reshaped: [B*c, patches, d_model]
      inputs_reshaped = torch.reshape(
        inputs,
        (
          inputs.shape[0] * inputs.shape[1],
          inputs.shape[2],
          inputs.shape[3],
        ),
      )

      # inputs_reshaped: [B*c, patches, d_model]
      inputs_reshaped = self.norm(inputs_reshaped)

      # put back data to the original shape
      inputs = torch.reshape(inputs_reshaped, inputs.shape)

    else:
      inputs = self.norm(inputs)

    return inputs




class TTMBatchNorm(nn.Module):
    def __init__(self, config: TTMConfiguration):
      super().__init__()
      self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)


    def forward(self, inputs: torch.Tensor):
      """
      Parameters:
        inputs (`torch.Tensor` of shape [B, sl, d_model]):
          input for Batch norm calculation
      Returns:
        `torch.Tensor` of shape [B, sl, d_model]
      """
      output = inputs.transpose(1, 2)  # output: [B, d_model, sl]
      output = self.batchnorm(output)
      return output.transpose(1, 2)