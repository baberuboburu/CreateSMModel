from src.architectures.base import BaseArchitecture, TemporalConvNet
from src.architectures.configuration import TCNConfiguration
from src.utils.early_stop import EarlyStopping
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


class TCN(BaseArchitecture):
  def __init__(self, config: TCNConfiguration):
    super().__init__()
    self.model = TemporalConvNet(config)
    self.model.to(config.backend)
    self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
    self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)  # 学習率スケジューラの設定

    # Prepare params
    self.ratio = config.ratio
    self.epoch = config.epoch
    self.learning_rate = config.learning_rate
    self.dropout = config.dropout
    self.kernel_size = config.kernel_size
    self.batch = config.batch
    self.sl = config.sl
    self.fl = config.fl
    self.train_size = config.train_size
    self.valid_size = config.valid_size
    self.test_size = config.test_size
    self.num_inputs = config.num_inputs
    self.num_outputs = config.num_outputs
    self.num_channels = config.num_channels
    self.min_delta = config.min_delta
    self.backend = config.backend
    self.predictions_dir = config.predictions_dir
    self.model_dir = config.model_dir
    self.model_name = config.model_name
  

  def train_DNPUs(self):
    # DNPUs
    df = self._prepare_data.setup_dnpus(ratio=self.ratio)
    train_loader, valid_loader, _ = self._prepare_data.dnpus_for_tcn(df, self.sl, self.fl, self.batch, train_size=self.train_size, valid_size=self.valid_size)

    losses = self.train(train_loader, valid_loader, self.num_inputs, self.sl, self.batch, self.epoch, self.model_dir, self.model_name)

    self._plot.plot_training_loss(losses, self.predictions_dir, self.epoch, d=0.05)
    
    return
  

  def test_DNPUs(self):
    # Prepare dataset
    df_normalized = self._prepare_data.setup_dnpus(ratio=self.ratio)
    _, _, test_loader = self._prepare_data.dnpus_for_tcn(df_normalized, self.sl, self.fl, self.batch, train_size=self.train_size)

    # Load model
    self.load(self.model_dir, self.model_name, self.backend)
    
    import time 
    start_time = time.time()

    all_targets, all_preds = self.test(test_loader, self.num_inputs, self.sl, self.batch)
    print(all_targets)
    print(all_preds)

    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    # Evaluation
    print(f'------- Evaluation (TCN) -------')
    loss_value = mean_squared_error(all_targets, all_preds)
    print(f'MSE: {loss_value}')

    # Plot
    self._plot.plot_predictions(all_targets, all_preds, self.predictions_dir, start=self.start_index, end=self.end_index)

    return
  

  def train(self, train_loader, valid_loader, num_inputs, sl, batch_size, epochs, tcn_model_dir, model_name):
    early_stopping = EarlyStopping(patience=5, min_delta=self.min_delta)
    losses = []

    for epoch in tqdm(range(epochs)):
      # Training
      self.model.train()

      for batch_idx, (data, target) in enumerate(train_loader):  # data.shape = [B × SL × n_inputs], target.shape = [B × FL × n_outputs]
        data, target = data.to(self.backend), target.to(self.backend)
        data = data.view(-1, num_inputs, sl)
        data, target = Variable(data), Variable(target)
        self.optimizer.zero_grad()
        output = self.model(data)

        output = output.squeeze().float()
        target = target.squeeze().float()

        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()

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
    
    # Save Model
    torch.save(self.model.state_dict(), os.path.join(tcn_model_dir, f'{model_name}.pt'))
    return losses


  def valid(self, valid_loader, num_inputs, sl):
    self.model.eval()
    valid_loss = 0

    with torch.no_grad():
      for data, target in valid_loader:
        data, target = data.to(self.backend), target.to(self.backend)
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
        data, target = data.to(self.backend), target.to(self.backend)
        data = data.view(-1, num_inputs, sl)
        data, target = Variable(data), Variable(target)
        pred = self.model(data)

        if len(target) == batch_size:
          target = target.cpu().numpy().reshape(batch_size)
          pred = pred.cpu().numpy().reshape(batch_size)
          all_targets = np.append(all_targets, target)
          all_preds = np.append(all_preds, pred)
      
      return all_targets, all_preds
  

  def load(self, model_dir: str, model_name: str, device: str):
    model_file_path = os.path.join(model_dir, f'{model_name}.pt')
    self.model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
    self.model.to(device)
    return self.model