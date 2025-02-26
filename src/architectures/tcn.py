from config.config import *
from src.architectures.base import BaseArchitecture
from src.utils.early_stop import EarlyStopping
from collections import OrderedDict
from typing import List
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.metrics import mean_squared_error


class TCN(BaseArchitecture):
  def __init__(self):
    super().__init__()
    self.model = TemporalConvNet(TCN_NUM_INPUTS, TCN_NUM_OUTPUTS, TCN_NUM_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT)
    self.optimizer = optim.Adam(self.model.parameters(), lr=TCN_LEARNING_RATE)
    self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)  # 学習率スケジューラの設定
  

  def train_DNPUs(self, ratio: float = 0.05, model_name: str = 'DNPUs'):
    # DNPUs
    df = self._prepare_data.setup_dnpus(ratio=1)
    train_loader, valid_loader, _ = self._prepare_data.dnpus_for_tcn(df, TCN_SL, TCN_FL, TCN_BATCH_SIZE, ratio=ratio)

    losses = self.train(train_loader, valid_loader, TCN_NUM_INPUTS, TCN_SL, TCN_BATCH_SIZE, TCN_EPOCHS, TCN_MODEL_DIR, model_name)

    self._plot.plot_training_loss(losses, PREDICTIONS_TCN_DIR, TCN_EPOCHS, d=0.05)
    
    return
  

  def test_DNPUs(self, ratio, model_name):
    # Prepare dataset
    df_normalized = self._prepare_data.setup_dnpus()
    _, _, test_loader = self._prepare_data.dnpus_for_tcn(df_normalized, TCN_SL, TCN_FL, TCN_BATCH_SIZE, ratio=ratio)

    # Load model
    self.load(TCN_MODEL_DIR, model_name, BACKEND)
    
    import time 
    start_time = time.time()

    all_targets, all_preds = self.test(test_loader, TCN_NUM_INPUTS, TCN_SL, TCN_BATCH_SIZE)
    print(all_targets)
    print(all_preds)

    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    # Evaluation
    print(f'------- Evaluation (TCN) -------')
    loss_value = mean_squared_error(all_targets, all_preds)
    print(f'MSE: {loss_value}')

    # Plot
    self._plot.plot_predictions(all_targets, all_preds, PREDICTIONS_TCN_DIR, start=self.start_index, end=self.end_index)

    return
  

  def train(self, train_loader, valid_loader, num_inputs, sl, batch_size, epochs, tcn_model_dir, model_name):
    early_stopping = EarlyStopping(patience=5, min_delta=TCN_MIN_DELTA)
    losses = []

    for epoch in tqdm(range(epochs)):
      # Training
      train_loss = 0
      self.model.train()

      for batch_idx, (data, target) in enumerate(train_loader):  # data.shape = [B × SL × n_inputs], target.shape = [B × FL × n_outputs]
        data = data.view(-1, num_inputs, sl)
        data, target = Variable(data), Variable(target)
        self.optimizer.zero_grad()
        output = self.model(data)

        output = output.squeeze().float()
        target = target.squeeze().float()

        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
        train_loss += loss
        if batch_idx == len(target):
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), train_loss.item()/len(target)))
          train_loss = 0

      # Validation
      val_loss = self.valid(valid_loader, num_inputs, sl)
      losses.append(val_loss)

      # Early stopping
      if early_stopping(val_loss):
        print(f"Training stopped at epoch {epoch}")
        break

      # Step scheduler (learning rate adjustment)
      self.scheduler.step()

      # Optionally, print the updated learning rate
      print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
    
    # Finish Training
    torch.save(self.model.state_dict(), os.path.join(tcn_model_dir, f'{model_name}.pt'))
    return losses


  def valid(self, valid_loader, num_inputs, sl):
    self.model.eval()
    valid_loss = 0

    with torch.no_grad():
      for data, target in valid_loader:
        data = data.view(-1, num_inputs, sl)
        data, target = Variable(data), Variable(target)
        pred = self.model(data)
        valid_loss += F.mse_loss(pred, target, reduction='mean').item()

      valid_loss /= len(valid_loader.dataset)
      print(f'\nValid set: Average loss: {valid_loss:.8f}')
      return valid_loss
  

  def test(self, test_loader, num_inputs, sl, batch_size):
    self.model.eval()
    all_targets = np.array([])
    all_preds = np.array([])

    with torch.no_grad():
      for data, target in test_loader:
        data = data.view(-1, num_inputs, sl)
        data, target = Variable(data), Variable(target)
        pred = self.model(data)

        if len(target) == batch_size:
          target = target.numpy().reshape(batch_size)
          pred = pred.numpy().reshape(batch_size)
          all_targets = np.append(all_targets, target)
          all_preds = np.append(all_preds, pred)
      
      return all_targets, all_preds
  

  def load(self, model_dir: str, model_name: str, device: str):
    model_file_path = os.path.join(model_dir, f'{model_name}.pt')
    self.model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
    self.model.to(device)
    return self.model


class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
  def __init__(
    self,
    num_inputs: int, num_outputs: int, num_channels: List[int],
    kernel_size: float, dropout: float
  ):

    super().__init__()
    self.layers = OrderedDict()
    self.num_levels = len(num_channels)
    for i in range(self.num_levels):
      dilation = 2 ** i
      n_in = num_inputs if (i == 0) else num_channels[i-1]
      n_out = num_channels[i]
      padding = (kernel_size - 1) * dilation
      # ========== TemporalBlock ==========
      self.layers[f'conv1_{i}'] = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation))
      self.layers[f'chomp1_{i}'] = Chomp1d(padding)
      self.layers[f'relu1_{i}'] = nn.ReLU()
      self.layers[f'dropout1_{i}'] = nn.Dropout(dropout)
      self.layers[f'conv2_{i}'] = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation))
      self.layers[f'chomp2_{i}'] = Chomp1d(padding)
      self.layers[f'relu2_{i}'] = nn.ReLU()
      self.layers[f'dropout2_{i}'] = nn.Dropout(dropout)
      self.layers[f'downsample_{i}'] = nn.Conv1d(n_in, n_out, 1) if (n_in != n_out) else None
      self.layers[f'relu_{i}'] = nn.ReLU()
      # ===================================
    self.relu = nn.ReLU()
    self.network = nn.Sequential(self.layers)
    self.linear = nn.Linear(num_channels[-1], num_outputs)


  def forward(self, x, debug=False):
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