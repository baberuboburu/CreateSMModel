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





class TransformerConfiguration():
  def __init__(self,
    ratio: float = None,
    fine_tune: bool = None,
    epoch: int = None,
    learning_rate: float = None, 
    dropout: float = None,
    batch: int = None,
    sl: int = None,
    fl: int = None,
    train_size: float = None,
    valid_size: float = None,
    test_size: float = None,
    num_inputs: int = None,
    num_outputs: int = None,
    num_layers: int = None,
    num_tunable_layers: int = None,
    num_heads: int = None,
    d_model: float = None,
    min_delta: float = None,
    backend: str = None,
    predictions_dir: str = None,
    model_dir: str = None,
    model_name: str = None,
    base_model_name: str = None,
  ):

    self.ratio = ratio if ratio else 0.05
    self.fine_tune = fine_tune if fine_tune else TRANSFORMER_FINE_TUNE
    self.epoch = epoch if epoch else TRANSFORMER_EPOCHS
    self.learning_rate = learning_rate if learning_rate else TRANSFORMER_LEARNING_RATE
    self.dropout = dropout if dropout else TRANSFORMER_DROPOUT
    self.batch = batch if batch else TRANSFORMER_BATCH_SIZE
    self.sl = sl if sl else TRANSFORMER_SL
    self.fl = fl if fl else TRANSFORMER_FL
    self.train_size = train_size if train_size else RATE_TRAINIG_DATASET
    self.valid_size = valid_size if valid_size else RATE_VALIDATION_DATASET
    self.test_size = test_size if test_size else RATE_TEST_DATASET
    self.num_inputs = num_inputs if num_inputs else TRANSFORMER_NUM_INPUTS
    self.num_outputs = num_outputs if num_outputs else TRANSFORMER_NUM_OUTPUTS
    self.num_layers = num_layers if num_layers else TRANSFORMER_NUM_LAYERS
    self.num_tunable_layers = num_tunable_layers if num_tunable_layers else TRANSFORMER_NUM_TUNABLE_LAYERS
    self.num_heads = num_heads if num_heads else TRANSFORMER_NUM_HEADS
    self.d_model = d_model if d_model else TRANSFORMER_D_MODEL
    self.min_delta = min_delta if min_delta else TRANSFORMER_MIN_DELTA
    self.backend = backend if backend else BACKEND
    self.predictions_dir = predictions_dir if predictions_dir else PREDICTIONS_TRANSFORMER_DIR
    self.model_dir = model_dir if model_dir else TRANSFORMER_MODEL_DIR
    self.model_name = model_name if model_name else datetime.now().strftime("%Y-%m-%d-%H-%M")
    self.base_model_name = base_model_name if base_model_name else f'FineTuned_{datetime.now().strftime("%Y-%m-%d-%H-%M")}'




class OnlyDecoderConfiguration():
  def __init__(self,
    ratio: float = None,
    epoch: int = None,
    learning_rate: float = None, 
    dropout: float = None,
    batch: int = None,
    sl: int = None,
    fl: int = None,
    train_size: float = None,
    valid_size: float = None,
    test_size: float = None,
    num_inputs: int = None,
    num_outputs: int = None,
    num_layers: int = None,
    num_heads: int = None,
    d_model: float = None,
    min_delta: float = None,
    backend: str = None,
    predictions_dir: str = None,
    model_dir: str = None,
    model_name: str = None,
  ):

    self.ratio = ratio if ratio else 0.05
    self.epoch = epoch if epoch else ONLY_DECODER_EPOCHS
    self.learning_rate = learning_rate if learning_rate else ONLY_DECODER_LEARNING_RATE
    self.dropout = dropout if dropout else ONLY_DECODER_DROPOUT
    self.batch = batch if batch else ONLY_DECODER_BATCH_SIZE
    self.sl = sl if sl else ONLY_DECODER_SL
    self.fl = fl if fl else ONLY_DECODER_FL
    self.train_size = train_size if train_size else RATE_TRAINIG_DATASET
    self.valid_size = valid_size if valid_size else RATE_VALIDATION_DATASET
    self.test_size = test_size if test_size else RATE_TEST_DATASET
    self.num_inputs = num_inputs if num_inputs else ONLY_DECODER_NUM_INPUTS
    self.num_outputs = num_outputs if num_outputs else ONLY_DECODER_NUM_OUTPUTS
    self.num_layers = num_layers if num_layers else ONLY_DECODER_NUM_LAYERS
    self.num_heads = num_heads if num_heads else ONLY_DECODER_NUM_HEADS
    self.d_model = d_model if d_model else ONLY_DECODER_D_MODEL
    self.min_delta = min_delta if min_delta else ONLY_DECODER_MIN_DELTA
    self.backend = backend if backend else BACKEND
    self.predictions_dir = predictions_dir if predictions_dir else PREDICTIONS_ONLY_DECODER_DIR
    self.model_dir = model_dir if model_dir else ONLY_DECODER_MODEL_DIR
    self.model_name = model_name if model_name else datetime.now().strftime("%Y-%m-%d-%H-%M")




class iTransformerConfiguration():
  def __init__(self,
    ratio: float = None,
    epoch: int = None,
    learning_rate: float = None, 
    dropout: float = None,
    batch: int = None,
    sl: int = None,
    fl: int = None,
    train_size: float = None,
    valid_size: float = None,
    test_size: float = None,
    num_inputs: int = None,
    num_outputs: int = None,
    num_layers: int = None,
    num_heads: int = None,
    d_model: float = None,
    min_delta: float = None,
    backend: str = None,
    predictions_dir: str = None,
    model_dir: str = None,
    model_name: str = None,
  ):

    self.ratio = ratio if ratio else 0.05
    self.epoch = epoch if epoch else ITRANSFORMER_EPOCHS
    self.learning_rate = learning_rate if learning_rate else ITRANSFORMER_LEARNING_RATE
    self.dropout = dropout if dropout else ITRANSFORMER_DROPOUT
    self.batch = batch if batch else ITRANSFORMER_BATCH_SIZE
    self.sl = sl if sl else ITRANSFORMER_SL
    self.fl = fl if fl else ITRANSFORMER_FL
    self.train_size = train_size if train_size else RATE_TRAINIG_DATASET
    self.valid_size = valid_size if valid_size else RATE_VALIDATION_DATASET
    self.test_size = test_size if test_size else RATE_TEST_DATASET
    self.num_inputs = num_inputs if num_inputs else ITRANSFORMER_NUM_INPUTS
    self.num_outputs = num_outputs if num_outputs else ITRANSFORMER_NUM_OUTPUTS
    self.num_layers = num_layers if num_layers else ITRANSFORMER_NUM_LAYERS
    self.num_heads = num_heads if num_heads else ITRANSFORMER_NUM_HEADS
    self.d_model = d_model if d_model else ITRANSFORMER_D_MODEL
    self.min_delta = min_delta if min_delta else ITRANSFORMER_MIN_DELTA
    self.backend = backend if backend else BACKEND
    self.predictions_dir = predictions_dir if predictions_dir else PREDICTIONS_ITRANSFORMER_DIR
    self.model_dir = model_dir if model_dir else ITRANSFORMER_MODEL_DIR
    self.model_name = model_name if model_name else datetime.now().strftime("%Y-%m-%d-%H-%M")