from config.config import *
from src.utils.process import Process
from src.utils.normalize import Normalize
from src.utils.plot import Plot
from src.utils.prepare_data import PrepareData
from src.utils.calculate import Calculate
from src.architectures.configuration import TCNConfiguration
from collections import OrderedDict
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


# Base
class BaseArchitecture():
  def __init__(self):
    self._process = Process()
    self._normalize = Normalize()
    self._plot = Plot()
    self._prepare_data = PrepareData()
    self._calculate = Calculate()
    
    self.start_index = 0
    self.end_index = 10000




# TCN
class Chomp1d(nn.Module):
  '''
  This class removes future information from data.
  '''
  def __init__(self, chomp_size):
    
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size


  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()




# TCN
class TemporalConvNet(nn.Module):
  def __init__(self, config: TCNConfiguration):
    super().__init__()
    self.to(config.backend)
    self.layers = OrderedDict()
    self.num_levels = len(config.num_channels)
    for i in range(self.num_levels):
      dilation = 2 ** i
      n_in = config.num_inputs if (i == 0) else config.num_channels[i-1]
      n_out = config.num_channels[i]
      padding = (config.kernel_size - 1) * dilation
      # ========== TemporalBlock ==========
      self.layers[f'conv1_{i}'] = weight_norm(nn.Conv1d(n_in, n_out, config.kernel_size, padding=padding, dilation=dilation))
      self.layers[f'chomp1_{i}'] = Chomp1d(padding)
      self.layers[f'relu1_{i}'] = nn.ReLU()
      self.layers[f'dropout1_{i}'] = nn.Dropout(config.dropout)
      self.layers[f'conv2_{i}'] = weight_norm(nn.Conv1d(n_out, n_out, config.kernel_size, padding=padding, dilation=dilation))
      self.layers[f'chomp2_{i}'] = Chomp1d(padding)
      self.layers[f'relu2_{i}'] = nn.ReLU()
      self.layers[f'dropout2_{i}'] = nn.Dropout(config.dropout)
      self.layers[f'downsample_{i}'] = nn.Conv1d(n_in, n_out, 1) if (n_in != n_out) else None
      self.layers[f'relu_{i}'] = nn.ReLU()
      # ===================================
    
    for key in self.layers:
      if self.layers[key] is not None:
        self.layers[key] = self.layers[key].to(config.backend)

    self.relu = nn.ReLU()
    self.network = nn.Sequential(self.layers)
    self.linear = nn.Linear(config.num_channels[-1], config.num_outputs)
    self.backend = config.backend


  def forward(self, x, debug=False):
    x = x.to(self.backend)
    self._debug_print(debug, '========== forward ==========')
    self._debug_print(debug, x.size())
    for i in range(self.num_levels):
      self._debug_print(debug, f'---------- block {i} ----------')
      self._debug_print(debug, 'in : ', x.size())
      # Residual Connection
      res = x if (self.layers[f'downsample_{i}'] is None) else self.layers[f'downsample_{i}'](x)
      out = self.layers[f'conv1_{i}'](x)
      out = self.layers[f'chomp1_{i}'](out)
      out = self.layers[f'relu1_{i}'](out)
      out = self.layers[f'dropout1_{i}'](out)
      out = self.layers[f'conv2_{i}'](out)
      out = self.layers[f'chomp2_{i}'](out)
      out = self.layers[f'relu2_{i}'](out)
      out = self.layers[f'dropout2_{i}'](out)
      self._debug_print(debug, 'out: ', out.size())
      self._debug_print(debug, 'res: ', res.size())
      x = self.layers[f'relu_{i}'](out + res)
      self._debug_print(debug, x.size())
      self._debug_print(debug, '-----------------------------')
    self._debug_print(debug, x.size())
    x = self.linear(x[:, :, -1])
    self._debug_print(debug, x.size())
    self._debug_print(debug, '=============================')
    return self.relu(x)
  

  def _debug_print(self, debug, *content):
    if debug:
      print(*content)