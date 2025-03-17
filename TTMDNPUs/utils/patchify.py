from TTMDNPUs.config.configuration import TTMConfiguration
import torch
import torch.nn as nn


class TTMPatchify(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()
    self.sl = config.sl
    self.patch_length = config.patch_length
    self.patch_stride = config.patch_stride

    if self.sl <= self.patch_length:
      raise ValueError(f"Sequence length ({self.sl}) has to be greater than the patch length ({self.patch_length})")

    # get the number of patches
    self.num_patches = (max(self.sl, self.patch_length) - self.patch_length) // self.patch_stride + 1
    new_sl = self.patch_length + self.patch_stride * (self.num_patches - 1)
    self.sequence_start = self.sl - new_sl


  def forward(self, past_values: torch.Tensor):
    """
    Parameters:
      past_values (`torch.Tensor` of shape `(B, sl, c)`, *required*):
        Input for patchification
    Returns:
      `torch.Tensor` of shape `(B, c, num_patches, patch_length)`
    """
    sl = past_values.shape[-2]
    if sl != self.sl:
      raise ValueError(f"Input sequence length ({sl}) doesn't match model configuration ({self.sl}).")

    output = past_values[:, self.sequence_start :, :]                                     # [B, new_sl, c]
    output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)  # [B, patches, c, patch_length]
    output = output.transpose(-2, -3).contiguous()                                        # [B, c, patches, patch_length] 
    return output