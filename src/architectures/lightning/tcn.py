from src.architectures.base import BaseArchitecture, TemporalConvNet
from src.architectures.configuration import TCNConfiguration
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


class TCNLightning(BaseArchitecture, L.LightningModule):
  def __init__(self, config: TCNConfiguration):
    # Setup architecture
    BaseArchitecture.__init__(self)
    L.LightningModule.__init__(self)
    self.model = TemporalConvNet(config)
    self.model.to(config.backend)
    self.loss_fn = nn.MSELoss()

    # Prepare dataset
    self.data_module = TCNDataModule(config)

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
    early_stopping = EarlyStopping(
      monitor="val_loss", 
      patience=5,
      min_delta=self.min_delta,
      mode="min"
    )
    trainer = L.Trainer(max_epochs=self.epoch, accelerator=self.backend, callbacks=[early_stopping], logger=False)
    
    # Training
    start_time = time.time()
    trainer.fit(self, self.data_module)
    end_time = time.time()
    print(f'Training Time: {end_time - start_time}:4f')

    # Save Model
    torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'{self.model_name}.pt'))
  

  def test_DNPUs(self):
    # Initialize
    self.test_step_outputs = []

    trainer = L.Trainer(max_epochs=self.epoch, accelerator=self.backend, logger=False)

    # Test
    start_time = time.time()
    trainer.test(self, self.data_module)
    end_time = time.time()
    print(f'Test Time: {(end_time - start_time):4f}')

    # Plot
    true_values = self.test_results["true_values"]
    prediction_values = self.test_results["prediction_values"]

    true_values = np.array(true_values)
    prediction_values = np.array(prediction_values)

    self._plot.plot_predictions(true_values, prediction_values, self.predictions_dir, start=self.start_index, end=self.end_index)


  def forward(self, x):
    return self.model(x)


  def training_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("train_loss", loss)
    return loss


  def validation_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("val_loss", loss, prog_bar=True)
    return loss


  def test_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("test_loss", loss, prog_bar=True)

    output_dict = {"true": target.cpu().numpy(), "pred": output.cpu().numpy()}
    self.test_step_outputs.append(output_dict)
    return output_dict


  def on_test_epoch_end(self):
      """
      各バッチの `test_step()` の戻り値を `self.test_step_outputs` から集約
      """
      true_values = []
      prediction_values = []

      for output in self.test_step_outputs:
        true_values.extend(output["true"])
        prediction_values.extend(output["pred"])

      self.test_results = {
        "true_values": np.array(true_values),
        "prediction_values": np.array(prediction_values)
      }

      self.test_step_outputs.clear()


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    return [optimizer], [scheduler]




class TCNDataModule(BaseArchitecture, L.LightningDataModule):
  def __init__(self, config: TCNConfiguration):
    BaseArchitecture.__init__(self)
    L.LightningDataModule.__init__(self)
    self.ratio = config.ratio
    self.batch_size = config.batch
    self.sl = config.sl
    self.fl = config.fl
    self.train_size = config.train_size
    self.valid_size = config.valid_size
    

  def setup(self, stage=None):
    df = self._prepare_data.setup_dnpus(ratio=self.ratio)
    self.train_loader, self.val_loader, self.test_loader = self._prepare_data.dnpus_for_tcn(
      df, self.sl, self.fl, self.batch_size, train_size=self.train_size, valid_size=self.valid_size
    )


  def train_dataloader(self):
    return self.train_loader


  def val_dataloader(self):
    return self.val_loader


  def test_dataloader(self):
    return self.test_loader