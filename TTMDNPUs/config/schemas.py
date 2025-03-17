from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from transformers.utils import ModelOutput


@dataclass
class OutputTTM(ModelOutput):
  loss: torch.FloatTensor = torch.tensor(0.0)
  prediction_outputs: torch.FloatTensor = None
  backbone_hidden_state: torch.FloatTensor = None
  loc: torch.FloatTensor = None
  scale: torch.FloatTensor = None


@dataclass
class OutputTTMBackBone(ModelOutput):
  '''
  backbone_hidden_state: [B, fl, c']
  patch_input:           [B, fl, c']
  loc:                   [B, fl, c']
  scale:                 [B, fl, c']
  '''
  backbone_hidden_state: torch.FloatTensor = None
  patch_input: torch.FloatTensor = None
  loc: Optional[torch.FloatTensor] = None
  scale: Optional[torch.FloatTensor] = None


@dataclass
class OutputTTMHead(ModelOutput):
  '''
  y_hat: [B, fl, c']
  '''
  y_hat: torch.FloatTensor = None


@dataclass
class OutputTTMOutput(ModelOutput):
  '''
  forecast: [B, fl, c']
  '''
  forecast: Tuple[torch.FloatTensor] = None


@dataclass
class OutputTTMEncoder(ModelOutput):
  '''
  encoder_output: [B, c, patch, d_model]
  '''
  encoder_output: torch.FloatTensor = None


@dataclass
class OutputTTMDecoder(ModelOutput):
  '''
  decoder_output: [B, c, patch, d_model]
  '''
  decoder_output: Tuple[torch.FloatTensor] = None


@dataclass
class OutputTTMTTMMixer(ModelOutput):
  '''
  '''
  embedding: torch.FloatTensor = None


@dataclass
class OutputTTMAdaptivePatchMixer(ModelOutput):
  '''
  '''
  hidden: torch.FloatTensor = None


@dataclass
class OutputTSMixer(ModelOutput):
  '''
  hidden: [B, c, patches, d_model]
  '''
  hidden: torch.FloatTensor = None


@dataclass
class OutputTSPatchMixer(ModelOutput):
  '''
  patch_mixed_hidden: [B, ?]
  '''
  patch_mixed_hidden: torch.FloatTensor = None


@dataclass
class OutputTSFeatureMixer(ModelOutput):
  '''
  feature_mixed_hidden: [B, c, patches, d_model]
  '''
  feature_mixed_hidden: torch.FloatTensor = None


@dataclass
class OutputTSChannelMixer(ModelOutput):
  '''
  channel_mixed_hidden: [B, c, patches, d_model]
  '''
  channel_mixed_hidden: torch.FloatTensor = None