from src.architectures.base import BaseArchitecture
from src.architectures.configuration import TransformerConfiguration
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


class TransformerLightning(BaseArchitecture, L.LightningModule):
  def __init__(self, config: TransformerConfiguration):
    # Setup architecture
    BaseArchitecture.__init__(self)
    L.LightningModule.__init__(self)
    self.model = TimeSeriesTransformer(config)

    # At fine tuning, load base model.
    if config.fine_tune:
      config.model_dir
      model_path = os.path.join(config.model_dir, f'{config.base_model_name}.pt')
      self.model.load_state_dict(torch.load(model_path))

    self.model.to(config.backend)
    self.loss_fn = nn.MSELoss()

    # Prepare dataset
    self.data_module = TransformerDataModule(config)

    # Prepare params
    self.ratio = config.ratio
    self.epoch = config.epoch
    self.learning_rate = config.learning_rate
    self.dropout = config.dropout
    self.batch = config.batch
    self.sl = config.sl
    self.train_size = config.train_size
    self.valid_size = config.valid_size
    self.test_size = config.test_size
    self.num_inputs = config.num_inputs
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
    print(f'Training Time: {(end_time - start_time):4f}')

    # Save Model
    torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'{self.model_name}.pt'))
  

  def test_DNPUs(self):
    # Initialize
    self.test_step_outputs = []

    # Load Model
    self.model.load_state_dict(torch.load(os.path.join(self.model_dir, f'{self.model_name}.pt')))

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

    self._plot.plot_fft(prediction_values, self.predictions_dir, prefix='predict')
    self._plot.plot_fft(true_values, self.predictions_dir, prefix='true')

    self._plot.plot_predictions(true_values, prediction_values, self.predictions_dir, start=self.start_index, end=self.end_index)


  def forward(self, x):
    return self.model(x)


  def training_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl).permute(0, 2, 1)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("train_loss", loss)
    return loss


  def validation_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl).permute(0, 2, 1)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("val_loss", loss, prog_bar=True)
    return loss


  def test_step(self, batch, batch_idx):
    data, target = batch
    data, target = data.to(self.backend), target.to(self.backend)
    data = data.view(-1, self.num_inputs, self.sl).permute(0, 2, 1)
    output = self.model(data).squeeze().float()
    target = target.squeeze().float()
    loss = F.mse_loss(output, target)
    self.log("test_loss", loss, prog_bar=True)

    output_dict = {"true": target.cpu().numpy(), "pred": output.cpu().numpy()}
    self.test_step_outputs.append(output_dict)
    return output_dict


  def on_test_epoch_end(self):
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




class TransformerDataModule(BaseArchitecture, L.LightningDataModule):
  def __init__(self, config: TransformerConfiguration):
    BaseArchitecture.__init__(self)
    L.LightningDataModule.__init__(self)
    self.ratio = config.ratio
    self.batch_size = config.batch
    self.sl = config.sl
    self.fl = config.fl
    self.train_size = config.train_size
    self.valid_size = config.valid_size
    self.num_inputs = config.num_inputs
    self._has_setup = False
    

  def setup(self, stage=None):
    if self._has_setup == True:
      return

    df = self._prepare_data.setup_dnpus(ratio=self.ratio)
    if self.num_inputs == 7:
      self.train_loader, self.val_loader, self.test_loader = self._prepare_data.dnpus_for_onlyF(
        df, self.sl, self.fl, self.batch_size, train_size=self.train_size, valid_size=self.valid_size
      )
    # if self.num_inputs == 7:
    #   self.test_loader, self.train_loader, self.val_loader = self._prepare_data.dnpus_for_onlyF(
    #     df, self.sl, self.fl, self.batch_size, train_size=self.train_size, valid_size=self.valid_size
    #   )
    elif self.num_inputs == 8:
      self.train_loader, self.val_loader, self.test_loader = self._prepare_data.dnpus_for_FT(
        df, self.sl, self.fl, self.batch_size, train_size=self.train_size, valid_size=self.valid_size
      )

    self._has_setup = True


  def train_dataloader(self):
    return self.train_loader


  def val_dataloader(self):
    return self.val_loader


  def test_dataloader(self):
    return self.test_loader




# # Transformer Model for Time-Series Forecasting
# class TimeSeriesTransformer(nn.Module):
#   def __init__(self, config: TransformerConfiguration):
#     super().__init__()
#     self.num_inputs = config.num_inputs
#     self.num_outputs = config.num_outputs
#     self.d_model = config.d_model
#     self.num_heads = config.num_heads
#     self.num_layers = config.num_layers
#     self.dropout = config.dropout
    
#     self.input_embedding = nn.Linear(self.num_inputs, self.d_model)
#     self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, self.d_model))
    
#     self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout)
#     self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
    
#     self.output_layer = nn.Linear(self.d_model, 1)


#   def forward(self, src):
#     src = self.input_embedding(src) + self.positional_encoding[:, :src.size(1), :]
    
#     memory = self.transformer_encoder(src)
#     output = self.output_layer(memory[:, -1, :])
    
#     return output



# class TimeSeriesTransformer(nn.Module):
#   def __init__(self, config: TransformerConfiguration):
#     super().__init__()

#     self.fine_tune = config.fine_tune
#     self.num_inputs = config.num_inputs
#     self.num_outputs = config.num_outputs
#     self.d_model = config.d_model
#     self.num_heads = config.num_heads
#     self.num_layers = config.num_layers
#     self.num_tunable_layers = config.num_tunable_layers
#     self.dropout = config.dropout

#     self.input_embedding = nn.Linear(self.num_inputs, self.d_model)
#     self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, self.d_model))

#     # Encoder
#     self.encoder_layers = nn.ModuleList([
#       nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout)
#       for _ in range(self.num_layers)
#     ])

#     self.output_layer = nn.Linear(self.d_model, 1)

#     # At fine tuning, only the last N layers are tunable. (N=self.num_tunable_layers)
#     if self.fine_tune:
#       for i, layer in enumerate(reversed(self.encoder_layers)):
#         requires_grad = i < self.num_tunable_layers
#         for param in layer.parameters():
#           param.requires_grad = requires_grad


#   def forward(self, src):
#     src = self.input_embedding(src) + self.positional_encoding[:, :src.size(1), :]
    
#     output = src
#     for layer in self.encoder_layers:
#       output = layer(output)

#     output = self.output_layer(output[:, -1, :])
#     return output






class TimeSeriesTransformer(nn.Module):
  def __init__(self, config: TransformerConfiguration):
    super().__init__()

    self.fine_tune = config.fine_tune
    self.num_inputs = config.num_inputs
    self.num_outputs = config.num_outputs
    self.d_model = config.d_model
    self.num_heads = config.num_heads
    self.num_layers = config.num_layers
    self.num_tunable_layers = config.num_tunable_layers
    self.dropout = config.dropout

    self.input_embedding = nn.Linear(self.num_inputs, self.d_model)
    self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, self.d_model))

    # Encoder
    self.encoder_layers = nn.ModuleList([
      nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout, batch_first=True)
      for _ in range(self.num_layers)
    ])

    self.output_layer = nn.Linear(self.d_model, 1)

    if self.fine_tune:
      for i, layer in enumerate(reversed(self.encoder_layers)):
        requires_grad = i < self.num_tunable_layers
        for param in layer.parameters():
          param.requires_grad = requires_grad

    # T-Fixup Initialization
    self._init_tfixup()


  def _init_tfixup(self):
    N = self.num_layers
    encoder_scale = 0.67 * (N ** -0.25)

    # Initialize input embedding (N(0, d^{-1/2}))
    init.normal_(self.input_embedding.weight, mean=0.0, std=self.d_model ** -0.5)
    init.zeros_(self.input_embedding.bias)

    # Initialize output embedding (Xavier + Scaling)
    init.xavier_normal_(self.output_layer.weight)
    self.output_layer.weight.data.mul_(encoder_scale)
    init.zeros_(self.output_layer.bias)

    # Apply to Linear in TransformerEncoderLayer
    for layer in self.encoder_layers:
      for sub_module in layer.modules():
        if isinstance(sub_module, nn.Linear):
          init.xavier_normal_(sub_module.weight)
          sub_module.weight.data.mul_(encoder_scale)
          if sub_module.bias is not None:
            init.zeros_(sub_module.bias)


  def forward(self, src):
    src = self.input_embedding(src) + self.positional_encoding[:, :src.size(1), :]

    output = src
    for layer in self.encoder_layers:
      output = layer(output)

    output = self.output_layer(output[:, -1, :])
    return output
