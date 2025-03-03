from config.config import *
from typing import List
from datetime import datetime


class TCNConfiguration():
  def __init__(self,
    ratio: float = None,
    epoch: int = None,
    learning_rate: float = None, 
    dropout: float = None,
    kernel_size: int = None,
    batch: int = None,
    sl: int = None,
    fl: int = None,
    train_size: float = None,
    valid_size: float = None,
    test_size: float = None,
    num_inputs: int = None,
    num_outputs: int = None,
    num_channels: List[int] = None,
    min_delta: float = None,
    backend: str = None,
    predictions_dir: str = None,
    model_dir: str = None,
    model_name: str = None,
  ):

    self.ratio = ratio if ratio else 0.05
    self.epoch = epoch if epoch else TCN_EPOCHS
    self.learning_rate = learning_rate if learning_rate else TCN_LEARNING_RATE
    self.dropout = dropout if dropout else TCN_DROPOUT
    self.kernel_size = kernel_size if kernel_size else TCN_KERNEL_SIZE
    self.batch = batch if batch else TCN_BATCH_SIZE
    self.sl = sl if sl else TCN_SL
    self.fl = fl if fl else TCN_FL
    self.train_size = train_size if train_size else RATE_TRAINIG_DATASET
    self.valid_size = valid_size if valid_size else RATE_VALIDATION_DATASET
    self.test_size = test_size if test_size else RATE_TEST_DATASET
    self.num_inputs = num_inputs if num_inputs else TCN_NUM_INPUTS
    self.num_outputs = num_outputs if num_outputs else TCN_NUM_OUTPUTS
    self.num_channels = num_channels if num_channels else TCN_NUM_CHANNELS
    self.min_delta = min_delta if min_delta else TCN_MIN_DELTA
    self.backend = backend if backend else BACKEND
    self.predictions_dir = predictions_dir if predictions_dir else PREDICTIONS_TCN_DIR
    self.model_dir = model_dir if model_dir else TCN_MODEL_DIR
    self.model_name = model_name if model_name else datetime.now().strftime("%Y-%m-%d-%H-%M")