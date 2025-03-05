from config.config import *
from src.architectures.base import BaseArchitecture
from originalTTM.TTM import TTM
from originalTTM.config.configuration import TTMConfiguration
import os
import math
from tsfm.tsfm_public.models.tinytimemixer.utils import count_parameters
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error


class OriginalTTM(BaseArchitecture):
  def __init__(self):
    super().__init__()


  # Zero-Shot Variant
  def zeroshot(self, new: bool = True):
    if new:
      zeroshot_trainer = self.zero_shot_train(
        TTM_MODEL_REVISION, TTM_LOSS, 
        TTM_FL, 
        TTM_MODEL_DIR, TTM_MODEL_ZEROSHOT_DIR, TTM_PER_DEVICE_EVAL_BATCH_SIZE
      )

    else:
      zeroshot_trainer = self.zero_shot_load(
        TTM_MODEL_DIR, TTM_MODEL_ZEROSHOT_DIR, TTM_PER_DEVICE_EVAL_BATCH_SIZE
      )

    return zeroshot_trainer
  

  # Few-Shot variant
  def fewshot(self, new: bool = True, ratio: float = 0.05, model_name: str = 'ttm/fewshots/DNPUs_8'):
    # 5% few-shot variant
    df_normalized = self._prepare_data.setup_dnpus(ratio=ratio)
    _, train_dataset, valid_dataset, _ = self._prepare_data.dnpus_for_ttm(df_normalized)

    if new:
      import time 
      start_time = time.time()
      fewshot_trainer = self.few_shot_train(
        train_dataset, valid_dataset, 
        TTM_MODEL_REVISION, TTM_LOSS, 
        TTM_FL, TTM_FEW_DROOPOUT, TTM_FEW_LEARNING_RATE, TTM_FEW_NUM_EPOCHS, 
        TTM_MODEL_DIR, TTM_MODEL_FEWSHOT_DIR, TTM_PER_DEVICE_EVAL_BATCH_SIZE, model_name
      )
      end_time = time.time()
      print(f'Training Time(Few {ratio * 100}%): {end_time - start_time}')

    else:
      fewshot_trainer = self.few_shot_load(
        TTM_MODEL_DIR, TTM_MODEL_FEWSHOT_DIR, TTM_PER_DEVICE_EVAL_BATCH_SIZE, model_name
      )

    return fewshot_trainer


  def DNPUs(self, trainer, type_: str, ratio: float = 1.0):
    df_normalized = self._prepare_data.setup_dnpus(ratio=ratio)
    _, _, _, test_dataset = self._prepare_data.dnpus_for_ttm(df_normalized)

    import time 
    start_time = time.time()
    all_targets, all_preds = self._calculate.calculate_predictions(
      trainer=trainer, 
      dataset=test_dataset, 
      start=self.start_index, 
      end=self.end_index,
      amplification=1.0
    )
    end_time = time.time()
    print(f'Prediction Time: {end_time - start_time}')

    self._plot.plot_predictions(
      all_targets, 
      all_preds, 
      PREDICTIONS_TTM_DIR, 
      plot_prefix=f'DNPUs_{type_}', 
      channel=0, 
      start=self.start_index, 
      end=self.end_index
    )

    # Evaluation
    print('------- Evaluation (Few-Shot) -------')
    mse = mean_squared_error(all_targets, all_preds)
    print(f'MSE: {mse}')


  def zero_shot_train(
    self, 
    ttm_version: str, 
    loss: str, 
    forecast_length: int, 
    ouput_dir: str, 
    model_dir: str, 
    per_device_eval_batch_size: int
    ):

    # Check if GPU is available and set the device accordingly
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(BACKEND)
    print(f"Using device: {device}")

    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2", revision=ttm_version, loss=loss, prediction_filter_length=forecast_length)

    zeroshot_trainer = Trainer(
      model=zeroshot_model,
      args=TrainingArguments(
        output_dir=ouput_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
      )
    )
    zeroshot_trainer.save_model(os.path.join(model_dir, f'ttm/zeroshot'))

    print(f'Params(Zero-Shot Model): {count_parameters(zeroshot_model)}')

    return zeroshot_trainer
  

  def zero_shot_load(self, ouput_dir: str, model_dir: str, per_device_eval_batch_size: int):
    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(os.path.join(model_dir, f'ttm/zeroshot'))

    zeroshot_trainer = Trainer(
      model=zeroshot_model,
      args=TrainingArguments(
        output_dir=ouput_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
      )
    )

    return zeroshot_trainer


  def few_shot_train(
    self, 
    train_dataset: Dataset, valid_dataset: Dataset, 
    ttm_version: str, 
    loss: str, 
    forecast_length: int, 
    head_dropout: float, 
    learning_rate: float,
    num_epochs: int, 
    output_dir: str, 
    model_dir: str, 
    per_device_eval_batch_size: int,
    model_name: str,
    ):

    # Check if GPU is available and set the device accordingly
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(BACKEND)
    print(f"Using device: {device}")

    fewshot_model = TinyTimeMixerForPrediction.from_pretrained(
      "ibm-granite/granite-timeseries-ttm-r2", 
      revision=ttm_version, loss=loss, 
      prediction_filter_length=forecast_length, 
      num_input_channels = len(COLUMN_INPUT),
      num_output_channels = len(COLUMN_TARGET),
      head_dropout=head_dropout,
      context_length=TTM_SL,
      patch_length=TTM_PATCH_SIZE,
      patch_stride=TTM_PATCH_STRIDE,
      ignore_mismatched_sizes=True  # Ignore the param shape of pretrained model.
    )

    print(
      "Number of params before freezing backbone",
      count_parameters(fewshot_model),
    )

    # fewshot_model のパラメータをモジュールごとに整理して表示
    from collections import defaultdict

    # モジュールごとのパラメータ数を集計
    param_counts = defaultdict(int)
    total_params = 0

    for name, param in fewshot_model.named_parameters():
      module_name = name.split('.')[0]  # 最上位のモジュール名を取得
      param_counts[module_name] += param.numel()
      total_params += param.numel()

    # 各モジュールごとのパラメータ数を表示
    print("Parameter distribution in fewshot_model:")
    for module, count in param_counts.items():
      print(f"{module}: {count:,} parameters")
    print(f"Total parameters: {total_params:,}")

    # Freeze the backbone of the model
    for param in fewshot_model.backbone.parameters():
      param.requires_grad = False

    # Count params
    print(
      "Number of params after freezing the backbone",
      count_parameters(fewshot_model),
    )

    # Move the model to the appropriate device
    fewshot_model.to(device)

    fewhsot_args = TrainingArguments(
      output_dir=os.path.join(model_dir, model_name),
      overwrite_output_dir=True,
      learning_rate=learning_rate,
      num_train_epochs=num_epochs,
      do_eval=True,                                             # True: do evaluation at learning using eval_dataset, False: no evaluation
      eval_strategy="epoch",                                    # 'no': no evaluation, 'epoch': evaluation after each epochs, 'steps': evaluation at each specified spte
      per_device_train_batch_size=per_device_eval_batch_size,   # Batch Size at Training
      per_device_eval_batch_size=per_device_eval_batch_size,    # Batch Size at Testing
      dataloader_num_workers=0,                                 # The number of threads downloading data
      report_to=None,                                           # Specify report tool (ex. wandb, tensorbord)
      save_strategy="epoch",                                    # Model saving timing
      logging_strategy="epoch",                                 # Logging timing
      save_total_limit=1,                                       # The number of saved model. (old one will be removed)
      logging_dir=os.path.join(output_dir, "logs"),             # Make sure to specify a logging directory
      load_best_model_at_end=True,                              # Load the best model when training ends
      metric_for_best_model="eval_loss",                        # Metric to monitor for early stopping
      greater_is_better=False,                                  # False: Lower loss is better, True: Higher accuracy is better
      fp16=torch.cuda.is_available(),                           # This value is False if you use the Apple silicon.
      # pin_memory=False                                        # Disable pin memory
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
      early_stopping_patience=5,     # Number of epochs with no improvement after which to stop
      early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(fewshot_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
      optimizer,
      learning_rate,
      epochs=num_epochs,
      steps_per_epoch=math.ceil(len(train_dataset) / (per_device_eval_batch_size)),
    )

    fewshot_trainer = Trainer(
      model=fewshot_model,
      args=fewhsot_args,
      train_dataset=train_dataset,
      eval_dataset=valid_dataset,
      callbacks=[early_stopping_callback, tracking_callback],
      optimizers=(optimizer, scheduler),
    )

    # Fine tune
    fewshot_trainer.train()

    # Save model
    fewshot_trainer.save_model(os.path.join(model_dir, model_name))

    return fewshot_trainer
  

  def pretrained_model(self, new: bool = True, ratio: float = 0.05, model_name: str = 'ttm/pretrained/DNPUs_8'):
    df_normalized = self._prepare_data.setup_dnpus(ratio=ratio)
    train_dataset, valid_dataset, _ = self._prepare_data.dnpus_for_original_ttm(df_normalized, ORIGINAL_TTM_SL, ORIGINAL_TTM_FL, ORIGINAL_TTM_BATCH_SIZE)

    if new:
      import time 
      start_time = time.time()
      pretrained_trainer = self.pretrained_model_train(
        train_dataset, valid_dataset, 
        len(COLUMN_INPUT), len(COLUMN_TARGET), ORIGINAL_TTM_SL, ORIGINAL_TTM_FL, 
        ORIGINAL_TTM_PATCH_SIZE, ORIGINAL_TTM_PATCH_STRIDE, ORIGINAL_TTM_PRETRAINED_DROOPOUT, ORIGINAL_TTM_PRETRAINED_LEARNING_RATE, ORIGINAL_TTM_PRETRAINED_NUM_EPOCHS, 
        ORIGINAL_TTM_MODEL_DIR, ORIGINAL_TTM_MODEL_PRETRAINED_DIR, ORIGINAL_TTM_PER_DEVICE_EVAL_BATCH_SIZE, model_name
      )
      end_time = time.time()
      print(f'Training Time(Few {ratio * 100}%): {end_time - start_time}')

    else:
      pretrained_trainer = self.pretrained_model_load(
        ORIGINAL_TTM_MODEL_DIR, ORIGINAL_TTM_MODEL_PRETRAINED_DIR, ORIGINAL_TTM_PER_DEVICE_EVAL_BATCH_SIZE, model_name
      )

    return pretrained_trainer


  def pretrained_model_train(
    self, 
    train_dataset: Dataset, valid_dataset: Dataset, 
    num_input_channels: int,
    num_output_channels: int,
    context_length: int,
    forecast_length: int, 
    patch_length: int,
    patch_stride: int,
    head_dropout: float, 
    learning_rate: float,
    num_epochs: int, 
    output_dir: str, 
    model_dir: str, 
    per_device_eval_batch_size: int,
    model_name: str,
    ):

    config = TTMConfiguration()
    config.num_input_channels = num_input_channels
    config.num_output_channels = num_output_channels
    config.head_dropout = head_dropout
    config.sl = context_length
    config.fl = forecast_length
    config.patch_length = patch_length
    config.patch_stride = patch_stride

    model = TTM(config=config)

    # pretrained_model のパラメータをモジュールごとに整理して表示
    from collections import defaultdict

    # モジュールごとのパラメータ数を集計
    param_counts = defaultdict(int)
    total_params = 0

    for name, param in model.named_parameters():
      module_name = name.split('.')[0]  # 最上位のモジュール名を取得
      param_counts[module_name] += param.numel()
      total_params += param.numel()

    # 各モジュールごとのパラメータ数を表示
    print("Parameter distribution in pretrained_model:")
    for module, count in param_counts.items():
      print(f"{module}: {count:,} parameters")
    print(f"Total parameters: {total_params:,}")

    pretrained_args = TrainingArguments(
      output_dir=os.path.join(model_dir, model_name),
      overwrite_output_dir=True,
      learning_rate=learning_rate,
      num_train_epochs=num_epochs,
      do_eval=True,                                             # True: do evaluation at learning using eval_dataset, False: no evaluation
      eval_strategy="epoch",                                    # 'no': no evaluation, 'epoch': evaluation after each epochs, 'steps': evaluation at each specified spte
      per_device_train_batch_size=per_device_eval_batch_size,   # Batch Size at Training
      per_device_eval_batch_size=per_device_eval_batch_size,    # Batch Size at Testing
      dataloader_num_workers=0,                                 # The number of threads downloading data
      report_to=None,                                           # Specify report tool (ex. wandb, tensorbord)
      save_strategy="epoch",                                    # Model saving timing
      logging_strategy="epoch",                                 # Logging timing
      save_total_limit=1,                                       # The number of saved model. (old one will be removed)
      logging_dir=os.path.join(output_dir, "logs"),             # Make sure to specify a logging directory
      load_best_model_at_end=True,                              # Load the best model when training ends
      metric_for_best_model="eval_loss",                        # Metric to monitor for early stopping
      greater_is_better=False,                                  # False: Lower loss is better, True: Higher accuracy is better
      fp16=torch.cuda.is_available(),                           # This value is False if you use the Apple silicon.
      # pin_memory=False                                        # Disable pin memory
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
      early_stopping_patience=5,     # Number of epochs with no improvement after which to stop
      early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
      optimizer,
      learning_rate,
      epochs=num_epochs,
      steps_per_epoch=math.ceil(len(train_dataset) / (per_device_eval_batch_size)),
    )

    pretrained_trainer = Trainer(
      model=model,
      args=pretrained_args,
      train_dataset=train_dataset,
      eval_dataset=valid_dataset,
      callbacks=[early_stopping_callback],
      optimizers=(optimizer, scheduler),
    )

    # Training
    pretrained_trainer.train()

    # Save model
    pretrained_trainer.save_model(os.path.join(model_dir, model_name))

    return pretrained_trainer


  def pretrained_model_load(self, ouput_dir: str, model_dir: str, per_device_eval_batch_size: int, model_name: str):
    model = TTM.from_pretrained(os.path.join(model_dir, model_name))

    trainer = Trainer(
      model=model,
      args=TrainingArguments(
        output_dir=ouput_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
      )
    )

    return trainer