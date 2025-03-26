from src.architectures.torch.ttm_dnpus import TTMDNPUs
from src.architectures.lightning.transformer import TransformerLightning
from TTMDNPUs.config.configuration import TTMConfiguration
from src.architectures.configuration import TCNConfiguration, TransformerConfiguration
import math
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


  def d_model_light(self):
    num_heads = 4
    d_models = []
    for i in range(1, self.n+1):
      if i:
        d_models.append((num_heads * i)**2)
    return d_models
  

  def num_encoder_layer(self):
    num_encoder_layers = []
    for i in range(2, self.n+2):
      if i:
        num_encoder_layers.append(i)
    return num_encoder_layers


  def num_decoder_layer(self):
    num_decoder_layers = []
    for i in range(2, self.n+2):
      if i:
        num_decoder_layers.append(i)
    return num_decoder_layers


  def dropout(self):
    dropouts = []
    def round_sig(x, sig=1):
      if x == 0:
        return 0
      return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    for i in range(1, self.n+1):
      if i:
        val = i * 0.1
        dropouts.append(round_sig(val, sig=1))
    return dropouts


instance = DefineGridParams(3)
# d_models = instance.d_model()
d_models = instance.d_model_light()
num_encoder_layers = instance.num_encoder_layer()
num_decoder_layers = instance.num_decoder_layer()
dropouts = instance.dropout()
print(d_models)
print(num_encoder_layers)
print(num_decoder_layers)
print(dropouts)


# # ttm_dnpus
# ttm_dnpus = TTMDNPUs()
# for d_model in d_models:
#   for num_encoder_layer in num_encoder_layers:
#     for num_decoder_layer in num_decoder_layers:
#       print(f'(d_model, encoder, decoder) = ({d_model}, {num_encoder_layer}, {num_decoder_layer})')

#       ttm_dnpus_config.d_model = d_model
#       ttm_dnpus_config.encoder_num_layers = num_encoder_layer
#       ttm_dnpus_config.decoder_num_layers = num_decoder_layer


# Transformer
ttm_dnpus_config = TTMConfiguration()
transformer_config = TransformerConfiguration()
for d_model in d_models:
  for num_encoder_layer in num_encoder_layers:
    for dropout in dropouts:
      print(f'(d_model, encoder, dropout) = ({d_model}, {num_encoder_layer}, {dropout})')

      transformer_config.d_model = d_model
      transformer_config.num_layers = num_encoder_layer
      transformer_config.ratio = 1
      transformer_config.model_name = f'F_100%_e={num_encoder_layer}_d={d_model}_dropout={dropout}'
      print(transformer_config.model_name)
      transformer_lightning = TransformerLightning(transformer_config)

      # start = time.time()
      # transformer_lightning.train_DNPUs()
      # end = time.time()
      # print(f'Training Time ({d_model}, {num_encoder_layer}, {dropout}): {(start - end):4f}')