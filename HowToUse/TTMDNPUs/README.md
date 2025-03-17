# About TTM Library
Original TTM needs not only features and also targets for training.  
The purpose of this libirary is to predict targets only from features as inputs.  


## How To Use
You can use this file as below.

```
from TTMDNPUs.TTM import TTM
from TTMDNPUs.config.configuration import TTMConfiguration
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

config = TTMConfiguration()

config.num_input_channels = num_input_channels
config.num_output_channels = num_output_channels
config.head_dropout = head_dropout
config.sl = context_length
config.fl = forecast_length
config.patch_length = patch_length
config.patch_stride = patch_stride

model = TTM(config=config)

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
  label_names=['future_target_values'],                     # To evaluate using 'eval_loss'
  greater_is_better=False,                                  # False: Lower loss is better, True: Higher accuracy is better
  fp16=torch.cuda.is_available(),                           # This value is False if you use the Apple silicon.
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
```


## Directory Structure
```
TTMDNPUs/
│── TTM.py                  # This is the main file. Included main, backbone, and head class.
│
├── config/                 # Configuration files
│   ├── configuration.py    # This file has default setting of this model.
│   ├── schamas.py          # Response schemas of main, backbone, head, and components class.
│
├── components/             # TTM DNPUs components. Included encoder, decoder, and so on.
│   ├── encoder.py          # Encoder Block.
│   ├── decoder.py          # Decoder Block.
│   ├── output.py           # Output Block. After decoder, this class is envoked.
│   ├── ttm_mixer.py        # Encoder and decoder use this component.
│   ├── ts_mixer.py         # TS Mixer Block. TTM Mixer uses this component.
│
├── utils/                  # Utility functions.
│   ├── gated_attention.py  # TTM Attention mechanism is Gated Attention.
│   ├── mlp.py              # MLP Block. This mlp has 2 layers.
│   ├── normalize.py        # Normalization class.
│   ├── patchify.py         # This file patchfy time series dataset.
│   ├── scaler.py           # Before calcurate loss function, predict value is scaled.
```
