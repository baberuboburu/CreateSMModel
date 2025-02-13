from config.config import *
from src.architectures.base import BaseArchitecture
from src.utils.early_stop import EarlyStopping
from timesfm.src.timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
from tqdm import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
from timesfm.src.timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error


class TimesFM(BaseArchitecture):
  def __init__(self):
    super().__init__()


  # Zero-Shot Variant
  def zeroshot(self):
    import time
    start_time = time.time()

    tfm_trainer = TimesFMTrainer()
    zeroshot_trainer = tfm_trainer.zero_shot(
      BACKEND, TIMESFM_SL, TIMESFM_NUM_LAYERS, TIMESFM_FL, 
      TIMESFM_INPUT_PATCH_LEN, TIMESFM_OUTPUT_PATCH_LEN, TIMESFM_MODEL_DIM,
      TIMESFM_PER_CORE_BATCH_SIZE, TIMESFM_HUGGINGFACE_REPO_ID
    )

    end_time = time.time()
    print(f'Training Time(Zero): {end_time - start_time}')

    return zeroshot_trainer


  # Few-Shot Variant
  def fewshot(self, new: bool, ratio: float = 0.10, model_name: str = 'fewshot'):
    # Load the pre trained model
    tfm = TimesFm(
      hparams=TimesFmHparams(
        context_len=TIMESFM_SL,
        horizon_len=TIMESFM_FL,
        input_patch_len=TIMESFM_INPUT_PATCH_LEN,
        # output_patch_len=TIMESFM_OUTPUT_PATCH_LEN,
        num_layers=TIMESFM_NUM_LAYERS,
        model_dims=TIMESFM_MODEL_DIM,
        backend=torch.device,
      ),
      checkpoint=TimesFmCheckpoint(
        huggingface_repo_id=TIMESFM_HUGGINGFACE_REPO_ID
      ),
    )
    tfm_trainer = TimesFMTrainer(pretrained_model=tfm, learning_rate=TIMESFM_LEARNING_RATE, weight_decay=TIMESFM_WEIGHT_DECAY, device=BACKEND)

    # Create new few-shot model
    if new:
      # Prepare dataset
      df_normalized = self._prepare_data.setup_dnpus()
      dataloader, _, _ = self._prepare_data.dnpus_for_timesfm(df_normalized, TIMESFM_BATCH_SIZE, TIMESFM_INPUT_PATCH_LEN, TIMESFM_OUTPUT_PATCH_LEN, TIMESFM_FL, ratio)
      # train_dataloader, valid_dataloader, _, _, _, _ = self._prepare_data.sample_for_timesfm("ettm1", BATCH_SIZE, FL, ratio)

      # Training
      import time
      start_time = time.time()
      
      fewshot_trainer, losses = tfm_trainer.few_shot_train(dataloader, epochs=TIMESFM_EPOCHS)

      end_time = time.time()
      print(f'Training Time(Few): {end_time - start_time}')

      # Plot MSE trends for epochs
      self._plot.plot_training_loss(losses, PREDICTIONS_TIMESFM_DIR, TIMESFM_EPOCHS, d=0.05)

      # Save a new model
      model_file_path = os.path.join(TIMESFM_MODEL_DIR, f'{model_name}.pt')
      torch.save(fewshot_trainer.state_dict(), model_file_path)
    
    # Use few-shot model
    else:
      fewshot_trainer = tfm_trainer.few_shot_load(TIMESFM_MODEL_DIR, model_name, BACKEND)

    return fewshot_trainer


  def DNPUs(self, trainer, type_: str, ratio: float = 1.0):
    # Prepare dataset
    df_normalized = self._prepare_data.setup_dnpus()
    _, _, test_data = self._prepare_data.dnpus_for_timesfm(df_normalized, TIMESFM_BATCH_SIZE, TIMESFM_INPUT_PATCH_LEN, TIMESFM_OUTPUT_PATCH_LEN, TIMESFM_FL, ratio)

    df_input = test_data.iloc[-(TIMESFM_SL+TIMESFM_FL):-TIMESFM_FL]
    df_true = test_data.iloc[-TIMESFM_FL:]

    train_tensor = torch.tensor(df_input[COLUMN_TARGET].values, dtype=torch.float)
    valid_tensor = torch.tensor(df_true[COLUMN_TARGET].values, dtype=torch.float)
    train_tensor = train_tensor.t()
    valid_tensor = valid_tensor.t()

    print(train_tensor.size())
    print(valid_tensor.size())

    # Get an index number from the output column
    column_list = list(test_data[COLUMN_TARGET].columns)
    column_index = column_list.index(COLUMN_TARGET[0]) if COLUMN_TARGET[0] in column_list else -1

    import time 
    start_time = time.time()

    # true_values, predicted_values
    frequency_input = [0] * train_tensor.size(0)
    predicted_values, _ = trainer.forecast(
      train_tensor,
      freq=frequency_input,
    )
    print(predicted_values.shape)

    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    # Evaluation
    print(f'------- Evaluation (Times FM - {type_} shot) -------')
    true_values = df_true[COLUMN_TARGET].to_numpy().reshape(-1, 1)
    loss_value = mean_squared_error(true_values, predicted_values[0])
    print(f'MSE: {loss_value}')

    # Plot
    forecast_tensor = torch.tensor(predicted_values)
    self._plot.plot_forecast(train_tensor, valid_tensor, forecast_tensor, TIMESFM_SL, TIMESFM_FL, PREDICTIONS_TIMESFM_DIR, f'DNPUs_{type_}', column_index)


  def ettm1(self, trainer, type_: str, ratio: float = 1.0):
    # Prepare dataset
    _, _, _, _, _, test_data = self._prepare_data.sample_for_timesfm("ettm1", TIMESFM_BATCH_SIZE, TIMESFM_FL, ratio)
    df_input = test_data.iloc[-(TIMESFM_SL+TIMESFM_FL):-TIMESFM_FL]
    df_true = test_data.iloc[-TIMESFM_FL:]

    target_column = ["OT"]
    channel_idx = 0

    train_tensor = torch.tensor(df_input[target_column].values, dtype=torch.float)
    valid_tensor = torch.tensor(df_true[target_column].values, dtype=torch.float)
    train_tensor = train_tensor.t()
    valid_tensor = valid_tensor.t()

    print(train_tensor.size())
    print(valid_tensor.size())

    # Get an index number from the output column
    column_list = list(test_data[target_column].columns)
    column_index = column_list.index(target_column[channel_idx]) if target_column[channel_idx] in column_list else -1

    import time 
    start_time = time.time()

    # true_values, predicted_values
    frequency_input = [0] * train_tensor.size(0)
    predicted_values, _ = trainer.forecast(
      train_tensor,
      freq=frequency_input,
    )
    print(predicted_values.shape)

    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    # Evaluation
    print(f'------- Evaluation (Times FM - {type_} shot) -------')
    true_values = test_data[target_column[channel_idx]]
    loss_value = mean_squared_error(true_values[:len(predicted_values[channel_idx])], predicted_values[channel_idx])
    print(f'MSE: {loss_value}')

    # Plot
    forecast_tensor = torch.tensor(predicted_values)
    self._plot.plot_forecast(train_tensor, valid_tensor, forecast_tensor, TIMESFM_SL, TIMESFM_FL, PREDICTIONS_TIMESFM_DIR, f'Ettm1_{type_}', column_index)


class TimesFMTrainer():
  def __init__(self, pretrained_model=None, learning_rate=1e-4, weight_decay=1e-4, device=BACKEND):
    if pretrained_model:
      self.device = device
      self.model = pretrained_model.to(self.device)
      self._freeze_parameters()
      self.optimizer = self._initialize_optimizer(learning_rate, weight_decay)
      self.criterion = nn.MSELoss()
      self.early_stopping = EarlyStopping(patience=5, min_delta=0.001)


  def zero_shot(
    self, 
    backend: str,
    sl: int, 
    num_layers: int,
    fl: int,
    input_patch_len: str, 
    output_patch_len: str, 
    model_dims: str, 
    per_core_batch_size: int,
    huggingface_repo_id: str
    ):

    # Initialize TimesFM model and read checkpoints
    zeroshot_trainer = TimesFm(
      hparams=TimesFmHparams(
        backend=backend,
        per_core_batch_size=per_core_batch_size,
        horizon_len=fl,
        num_layers=num_layers,
        context_len=sl,
        input_patch_len=input_patch_len,
        output_patch_len=output_patch_len,
        model_dims=model_dims,
      ),
      checkpoint=TimesFmCheckpoint(
        huggingface_repo_id=huggingface_repo_id
      ),
    )

    return zeroshot_trainer


  def few_shot_train(self, dataloader, epochs=10):
    scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
    losses = []
    avg_loss = 0.0

    for epoch in tqdm(range(epochs)):
      self.model.train()
      total_loss = 0.0
      for idx, (inputs, targets) in enumerate(dataloader):
        outputs = self.model(inputs, 'train')
        outputs = outputs.mean(dim=1)    # (B, Col_inputs, P_output, 10) → (B, P_output, 10) (※ Col_outputs=1)
        outputs = outputs.mean(dim=-1)   # (32, 128, 10) → (32, 128)
        outputs = outputs.unsqueeze(-1)  # (32, 128) → (32, 128, 1)

        if idx % 20 == 0:
          print(f'({epoch}, {idx}): {avg_loss}')

        loss = self.criterion(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        scheduler.step()

        total_loss += loss.item()
      
      avg_loss = total_loss / (idx + 1)
      losses.append(avg_loss)
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

      if self.early_stopping(total_loss):
        break

      scheduler.step()
    

    return self.model, losses


  def few_shot_load(self, model_dir: str, model_name: str, device: str):
    model_file_path = os.path.join(model_dir, f'{model_name}.pt')
    self.model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
    self.model.to(device)
    return self.model


  def _freeze_parameters(self):
    total_params = sum(p.numel() for p in self.model.parameters())  # 総パラメータ数
    frozen_params = 0
    trainable_params = 0

    # Freeze parameters 'not' included "horizon_ff_layer"
    for param in self.model._model.parameters():
      param.requires_grad = False
      frozen_params += param.numel()  # Count Frozen parameters

    for name, param in self.model.named_parameters():
      if any(x in name for x in ["horizon_ff_layer", "output_layer"]):
        param.requires_grad = True
        trainable_params += param.numel()
    
    frozen_params -= trainable_params

    # Result
    print(f"Total parameters: {total_params}")
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {trainable_params}")


  def _initialize_optimizer(self, learning_rate, weight_decay):
    return optim.AdamW(
      filter(lambda p: p.requires_grad, self.model.parameters()),
      lr=learning_rate,
      weight_decay=weight_decay
    )