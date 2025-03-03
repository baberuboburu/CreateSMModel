from config.config import *
from src.architectures.base import BaseArchitecture
from src.utils.early_stop import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class FineTuneStaticSM(BaseArchitecture):
  def __init__(self):
    super().__init__()
    self.model = StaticSurrogateModel()
    checkpoint = self.load('surrogate_model.pt', BACKEND)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer = optim.Adam(self.model.parameters(), lr=TCN_LEARNING_RATE)
    self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)  # 学習率スケジューラの設定
  


  def train_DNPUs(self, ratio: float = 0.05, model_name: str = 'DNPUs'):
    # DNPUs
    df = self._prepare_data.setup_dnpus(ratio=1)
    train_loader, valid_loader, _ = self._prepare_data.dnpus_for_static(df, STATIC_SL, STATIC_FL, STATIC_BATCH_SIZE, ratio=ratio)

    losses = self.train(train_loader, valid_loader, STATIC_BATCH_SIZE, STATIC_EPOCHS, STATIC_MODEL_DIR, model_name)

    self._plot.plot_training_loss(losses, PREDICTIONS_STATIC_DIR, STATIC_EPOCHS, d=0.05)
    
    return


  def test_DNPUs(self, ratio, model_name):
    # Prepare dataset
    df_normalized = self._prepare_data.setup_dnpus()
    _, _, test_loader = self._prepare_data.dnpus_for_tcn(df_normalized, STATIC_SL, STATIC_FL, STATIC_BATCH_SIZE, ratio=ratio)

    # Load model
    self.load(model_name, BACKEND)
    
    import time 
    start_time = time.time()

    all_targets, all_preds = self.test(test_loader, STATIC_BATCH_SIZE)
    print(all_targets)
    print(all_preds)

    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    # Evaluation
    print(f'------- Evaluation (Fine Tuned Model Using Static Model) -------')
    loss_value = mean_squared_error(all_targets, all_preds)
    print(f'MSE: {loss_value}')

    # Plot
    self._plot.plot_predictions(all_targets, all_preds, PREDICTIONS_STATIC_DIR, start=self.start_index, end=self.end_index)

    return


  def unfreeze_parameters(self):
    param_tunable = 0
    param_fixed = 0

    for name, param in self.model.named_parameters():
      if 'raw_model.10' in name or 'raw_model.8' in name:
        param.requires_grad = True
        param_tunable += param.numel()
      else:
        param.requires_grad = False
        param_fixed += param.numel()

    print(f"Frozen Parameter: {param_fixed} \nTunable Parameter: {param_tunable}")


  def train(self, train_loader, valid_loader, batch_size, epochs, static_model_dir, model_name):
    # Define fixed parameter and tunable parameter for fine tuning
    self.unfreeze_parameters()

    early_stopping = EarlyStopping(patience=5, min_delta=STATIC_MIN_DELTA)
    losses = []

    for epoch in tqdm(range(epochs)):
      # Training
      train_loss = 0
      self.model.train()

      for batch_idx, (data, target) in enumerate(train_loader):
        data = data.squeeze(1)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
        train_loss += loss
        if batch_idx == len(train_loader) - 1:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), train_loss.item()/len(target)))
          train_loss = 0

      # Validation
      val_loss = self.valid(valid_loader)
      losses.append(val_loss)

      # Early stopping
      if early_stopping(val_loss):
        print(f"Training stopped at epoch {epoch}")
        break

      # 学習率の更新
      self.scheduler.step()

      # 損失の表示
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {val_loss:.8f}")

    # Finish Training
    torch.save(self.model.state_dict(), os.path.join(static_model_dir, f'{model_name}.pt'))
    return losses


  def valid(self, valid_loader):
    self.model.eval()
    valid_loss = 0

    with torch.no_grad():
      for data, target in valid_loader:
        data = data.squeeze(1)
        data, target = Variable(data), Variable(target)
        pred = self.model(data)
        valid_loss += F.mse_loss(pred, target, reduction='mean').item()

      valid_loss /= len(valid_loader)
      print(f'\nValid set: Average loss: {valid_loss:.8f}')
      return valid_loss


  def test(self, test_loader, batch_size):
    self.model.eval()
    all_targets = np.array([])
    all_preds = np.array([])

    with torch.no_grad():
      for data, target in test_loader:
        data = data.squeeze(1)
        data, target = Variable(data), Variable(target)
        pred = self.model(data)

        if len(target) == batch_size:
          target = target.numpy().reshape(batch_size)
          pred = pred.numpy().reshape(batch_size)
          all_targets = np.append(all_targets, target)
          all_preds = np.append(all_preds, pred)
      
      return all_targets, all_preds


  def load(self, model_name: str, backend: str):
    checkpoint = torch.load(os.path.join(STATIC_MODEL_DIR, model_name), map_location=torch.device(backend), weights_only=False)
    return checkpoint


class StaticSurrogateModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.raw_model = nn.Sequential(
      nn.Linear(7, STATIC_NUM_LAYERS[0]),
      nn.ReLU(),
      nn.Linear(STATIC_NUM_LAYERS[0], STATIC_NUM_LAYERS[1]),
      nn.ReLU(),
      nn.Linear(STATIC_NUM_LAYERS[1], STATIC_NUM_LAYERS[2]),
      nn.ReLU(),
      nn.Linear(STATIC_NUM_LAYERS[2], STATIC_NUM_LAYERS[3]),
      nn.ReLU(),
      nn.Linear(STATIC_NUM_LAYERS[3], STATIC_NUM_LAYERS[4]),
      nn.ReLU(),
      nn.Linear(STATIC_NUM_LAYERS[4], 1),
    )

  def forward(self, x):
    return self.raw_model(x)