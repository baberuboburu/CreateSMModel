from originalTTM.config.configuration import TTMConfiguration
from originalTTM.config.schemas import OutputTTMBackBone, OutputTTMHead
from originalTTM.components.encoder import TTMEncoder
from originalTTM.components.decoder import TTMDecoder
from originalTTM.components.output import TTMOutput
from originalTTM.utils.scaler import TTMScaler
from originalTTM.utils.patchify import TTMPatchify
from typing import Optional
import torch
import torch.nn as nn
# from transformers.modeling_utils import PreTrainedModel


class TTM(nn.Module):
  def __init__(self, config: TTMConfiguration):
    super().__init__()

    self.backbone = TTMBackBone(config)
    self.head = TTMHead(config)

    self.loss = nn.MSELoss(reduction="mean")
    # self.loss = nn.L1Loss()
  

  def forward(self,
    past_values: torch.Tensor,
    future_values: torch.Tensor = None,
    observed_mask: torch.Tensor = None,
  ):

    # Backbone
    backbone_output = self.backbone(
      past_values,
      observed_mask=observed_mask
    )

    decoder_input = backbone_output.backbone_hidden_state

    # Output
    y_hat = self.head(decoder_input).y_hat  # [B, fl, c']

    # Instance Normalization
    loc = backbone_output.loc
    scale = backbone_output.scale
    y_hat = y_hat * scale + loc
    print(y_hat.shape)
    print(future_values.shape)


    if future_values is not None and self.loss is not None:
      # y_hat = y_hat.to("cpu")
      # future_values = future_values.to("cpu")
      loss_val = self.loss(y_hat, future_values)
      print(loss_val)
    else:
      loss_val = None

    return {
      "loss": loss_val,
      "prediction_outputs": y_hat,
      "backbone_hidden_state": backbone_output.backbone_hidden_state,
      "loc": loc,
      "scale": scale,
    }



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