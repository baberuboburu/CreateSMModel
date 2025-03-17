from src.architectures.torch.ttm_dnpus import TTMDNPUs
from TTMDNPUs.config.configuration import TTMConfiguration
import time


class DefineGridParams():
  def __init__(self, n):
    self.n = n


  def d_model(self):
    d_origin = 7
    d_models = []
    for i in range(0, d_origin*self.n*2+1, d_origin*2):
      if i:
        d_models.append(d_origin * i)
    return d_models
  

  def num_encoder_layer(self):
    num_encoder_layers = []
    for i in range(4, self.n+3):
      if i:
        num_encoder_layers.append(i)
    return num_encoder_layers


  def num_decoder_layer(self):
    num_decoder_layers = []
    for i in range(2, self.n+1):
      if i:
        num_decoder_layers.append(i)
    return num_decoder_layers


instance = DefineGridParams(4)
d_models = instance.d_model()
num_encoder_layers = instance.num_encoder_layer()
num_decoder_layers = instance.num_decoder_layer()
print(d_models)
print(num_encoder_layers)
print(num_decoder_layers)


ttm_dnpus = TTMDNPUs()
config = TTMConfiguration()

for d_model in d_models:
  for num_encoder_layer in num_encoder_layers:
    for num_decoder_layer in num_decoder_layers:
      config.d_model = d_model
      config.encoder_num_layers = num_encoder_layer
      config.decoder_num_layers = num_decoder_layer
      # ttm_dnpus.
      # print(f'(d_model, encoder, decoder) = ({d_model}, {num_encoder_layer}, {num_decoder_layer})')