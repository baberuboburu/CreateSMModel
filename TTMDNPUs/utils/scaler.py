from TTMDNPUs.config.configuration import TTMConfiguration
import torch
import torch.nn as nn


class TTMScaler(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    if config.scaler_type == 'mean':
      self.scaler = TTMMeanScaler(config)
    elif config.scaler_type == 'std':
      self.scaler = TTMStdScaler(config)
    else:
      self.scaler = TTMNOPScaler(config)
  

  def forward(self, past_values, observed_mask):
    return self.scaler(past_values, observed_mask)




class TTMMeanScaler(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.dim = config.scaling_dim
    self.keepdim = config.scaling_keepdim
    self.minimum_scale = config.minimum_scale


  def forward(self, data: torch.Tensor, observed_mask: torch.Tensor):
    """
    Parameters:
      data (`torch.Tensor` of shape [B, sl, c]):
        input for Batch norm calculation
      observed_mask (`torch.BoolTensor` of shape [B, sl, c]):
        Calculating the scale on the observed indicator.
    Returns:
      tuple of `torch.Tensor` of shapes
        ([B, sl, c],[B, 1, c], [B, 1, c])
    """
    ts_sum = (data * observed_mask).abs().sum(self.dim, keepdim=True)
    num_observed = observed_mask.sum(self.dim, keepdim=True)

    scale = ts_sum / torch.clamp(num_observed, min=1)

    batch_sum = ts_sum.sum(dim=0)
    batch_observations = torch.clamp(num_observed.sum(0), min=1)
    default_scale = torch.squeeze(batch_sum / batch_observations)

    # apply default scale where there are no observations
    scale = torch.where(num_observed > 0, scale, default_scale)

    # ensure the scale is at least `self.minimum_scale`
    scale = torch.clamp(scale, min=self.minimum_scale)
    scaled_data = data / scale

    if not self.keepdim:
      scale = scale.squeeze(dim=self.dim)

    return scaled_data, torch.zeros_like(scale), scale




class TTMStdScaler(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.dim = config.scaling_dim
    self.keepdim = config.scaling_keepdim
    self.minimum_scale = config.minimum_scale
    self.num_input_channels = config.num_input_channels
    self.num_output_channels = config.num_output_channels


  def forward(self, data: torch.Tensor, observed_mask: torch.Tensor):
    """
    Parameters:
      data (`torch.Tensor` of shape [B, sl, c]):
        input for Batch norm calculation
      observed_mask (`torch.BoolTensor` of shape [B, sl, c]):
        Calculating the scale on the observed indicator.
    Returns:
        tuple of `torch.Tensor` of shapes
          ([B, sl, c],[B, 1, c],[B, 1, c])
    """
    denominator = observed_mask.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0)
    
    loc = (data * observed_mask).sum(self.dim, keepdim=self.keepdim) / denominator  # [B, 1, c]

    variance = (((data - loc) * observed_mask) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
    scale = torch.sqrt(variance + self.minimum_scale)  # [B, 1, c]

    scaled_data = (data - loc) / scale

    # c != c' => Select max mean and scale
    if self.num_input_channels != self.num_output_channels:
      max_loc, max_idx = loc.max(dim=-1, keepdim=True)
      scale_max = torch.gather(scale, dim=-1, index=max_idx)      # [B, 1, 1]
      
      loc = max_loc.expand(-1, -1, self.num_output_channels)      # [B, 1, c']
      scale = scale_max.expand(-1, -1, self.num_output_channels)  # [B, 1, c']


    return scaled_data, loc, scale




class TTMNOPScaler(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()
    self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
    self.keepdim = config.keepdim if hasattr(config, "scaling_keepdim") else True


  def forward(self, data: torch.Tensor, observed_mask: torch.Tensor = None):
    """
    Parameters:
      data (`torch.Tensor` of shape [B, sl, c]):
        input for Batch norm calculation
    Returns:
      tuple of `torch.Tensor` of shapes
        ([B, sl, c],[B, 1, c],[B, 1, c])
    """
    scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
    loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
    return data, loc, scale