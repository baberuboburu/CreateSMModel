from TTMDNPUs.config.configuration import TTMConfiguration
from TTMDNPUs.config.schemas import OutputTTM, OutputTTMBackBone, OutputTTMHead
from TTMDNPUs.components.encoder import TTMEncoder
from TTMDNPUs.components.decoder import TTMDecoder
from TTMDNPUs.components.output import TTMOutput
from TTMDNPUs.utils.scaler import TTMScaler
from TTMDNPUs.utils.patchify import TTMPatchify
from typing import Optional
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel


class TTM(PreTrainedModel):
  config_class = TTMConfiguration()


  def __init__(self, config: TTMConfiguration):
    super().__init__(config)

    self.backbone = TTMBackBone(config)
    self.head = TTMHead(config)

    self.loss = nn.MSELoss(reduction="mean")

    self.cnn2 = nn.Sequential(
      nn.Conv1d(
        in_channels=config.num_input_channels, 
        out_channels=4, 
        kernel_size=3, 
        padding=1
      ),
      nn.ReLU(),
      nn.Conv1d(
        in_channels=4, 
        out_channels=config.num_output_channels, 
        kernel_size=3, 
        padding=1
      ),
      nn.ReLU()
    )
  

  def forward(self,
    past_feature_values: torch.Tensor,
    future_target_values: torch.Tensor = None,
    observed_mask: torch.Tensor = None,
  ):
    # [B, sl, c] → [B, sl, c']
    past_feature_values = past_feature_values.permute(0, 2, 1)
    past_feature_values = self.cnn2(past_feature_values)
    past_feature_values = past_feature_values.permute(0, 2, 1)

    # Backbone
    backbone_output = self.backbone(
      past_feature_values,
      observed_mask=observed_mask
    )

    decoder_input = backbone_output.backbone_hidden_state

    # Output
    y_hat = self.head(decoder_input).y_hat  # [B, fl, c']

    # Instance Normalization
    loc = backbone_output.loc
    scale = backbone_output.scale
    y_hat = y_hat * scale + loc


    if future_target_values is not None and self.loss is not None:
      # y_hat = y_hat.to("cpu")
      # future_target_values = future_target_values.to("cpu")
      loss_val = self.loss(y_hat, future_target_values)
    else:
      loss_val = None

    return OutputTTM(
      loss=loss_val,
      prediction_outputs=y_hat,
      backbone_hidden_state=backbone_output.backbone_hidden_state,
      loc=loc,
      scale=scale
    )



class TTMBackBone(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.encoder = TTMEncoder(config)
    self.scaler = TTMScaler(config)
    self.patchify = TTMPatchify(config)
  

  def forward(
    self,
    past_values: torch.Tensor,
    observed_mask: Optional[torch.Tensor] = None,
  ):

    if observed_mask is None:
      observed_mask = torch.ones_like(past_values)
    scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

    patched_x = self.patchify(scaled_past_values)  # [B, c, patches, patch_length]

    enc_input = patched_x

    encoder_output = self.encoder(enc_input)
    # encoder_output = TinyTimeMixerEncoderOutput(*encoder_output)

    return OutputTTMBackBone(
      backbone_hidden_state=encoder_output.encoder_output,
      patch_input=patched_x,
      loc=loc,
      scale=scale,
    )



class TTMHead(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.decoder = TTMDecoder(config)
    self.output = TTMOutput(config)


  def forward(self, decoder_input):

    # Decoder
    decoder_output = self.decoder(decoder_input).decoder_output

    # Predict Result
    y_hat = self.output(decoder_output).forecast

    return OutputTTMHead(y_hat=y_hat)