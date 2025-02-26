from config.config import *
from src.utils.process import Process
from src.utils.normalize import Normalize
from src.utils.plot import Plot
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class PrepareData():
  def __init__(self):
    self._process = Process()
    self._normalize = Normalize()
    self._plot = Plot()
    
    self.start_index = 0
    self.end_index = 10000
  

  def setup_dnpus(self, ratio: float = 1, normalized_type: str = 'min_max'):
    # Prepare data
    df = self._process.load_and_process_data(DNPUs_DATA, len(COLUMN_OBSERVABLE), len(COLUMN_TARGET))

    # # Plot Time Series
    # self._plot.plot_time_series(df.head(15000), len(COLUMN_OBSERVABLE), len(COLUMN_TARGET), ANALYSIS_DNPUs_DIR)
    # self._plot.plot_original_time_series(df, ANALYSIS_DNPUs_DIR)
    # self._plot.plot_decompose_time_series(df, ANALYSIS_DNPUs_DIR)
    # self._plot.plot_stationarity(df.head(10000))
    # self._plot.plot_acf_and_pacf(df.head(10000), ANALYSIS_DNPUs_DIR)

    # extract {ratio * 100}% dataset (Used only at Few-shot variant)
    print(f'Full Data Shape (Before): {df.shape}')
    df = df.head(int(df.shape[0] * ratio))
    print(f'Full Data Shape (After): {df.shape}')

    # normalization
    if normalized_type == 'min_max':
      df_normalized = self._normalize.min_max_normalize(df, COLUMN_OBSERVABLE, COLUMN_TARGET)
    elif normalized_type == 'instance':
      df_normalized = self._normalize.instance_normalize(df, COLUMN_OBSERVABLE, COLUMN_TARGET)
    elif normalized_type == 'batch':
      df_normalized = self._normalize.batch_normalize(df, COLUMN_OBSERVABLE, COLUMN_TARGET)

    return df_normalized


  def dnpus_for_ttm(self, df_normalized: pd.DataFrame):
    # Split the dataset
    full_dataset, train_dataset, valid_dataset, test_dataset = self._process.prepare_learning_dataset(
      df_normalized, 
      RATE_TRAINIG_DATASET, RATE_TEST_DATASET,
      TTM_SL, TTM_FL,
      timestamp_column=COLUMN_TIMESTAMP, id_columns=COLUMN_ID, target_columns=COLUMN_TARGET, observable_columns=COLUMN_OBSERVABLE, control_columns=COLUMN_CONTROL
    )

    print(f"Data length: full = {len(full_dataset)}")
    print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

    return full_dataset, train_dataset, valid_dataset, test_dataset
    
  
  def sample_for_timesfm(self, dataset: str, batch_size: int, fl: int, ratio: float = 1.0):
    DATA_DICT = {
      "ettm1": {"boundaries": [34560, 46080, 57600], "data_path": "./timesfm/datasets/ETT-small/ETTm1.csv", "freq": "15min"},
      "ettm2": {"boundaries": [34560, 46080, 57600], "data_path": "./timesfm/datasets/ETT-small/ETTm2.csv", "freq": "15min"},
      "etth2": {"boundaries": [8640, 11520, 14400], "data_path": "./timesfm/datasets/ETT-small/ETTh2.csv", "freq": "H"},
      "etth1": {"boundaries": [8640, 11520, 14400], "data_path": "./timesfm/datasets/ETT-small/ETTh1.csv", "freq": "H"},
      "elec": {"boundaries": [18413, 21044, 26304], "data_path": "./timesfm/datasets/electricity/electricity.csv", "freq": "H"},
      "traffic": {"boundaries": [12280, 14036, 17544], "data_path": "./timesfm/datasets/traffic/traffic.csv", "freq": "H"},
      "weather": {"boundaries": [36887, 42157, 52696], "data_path": "./timesfm/datasets/weather/weather.csv", "freq": "10min"},
    }

    data_path = DATA_DICT[dataset]["data_path"]
    data_df = pd.read_csv(data_path).iloc[:, 1:]
    target_column = ['OT']
    n = int((len(data_df)*ratio) - batch_size) // batch_size
    print(int((len(data_df)*ratio)))
    print(n)

    train_data_start = 0
    train_data_end = batch_size*n
    valid_data_start = TIMESFM_SL
    valid_data_end = TIMESFM_SL + batch_size*(n-1) + fl
    test_data_start = valid_data_end + 1
    test_data_end = valid_data_end + self.end_index

    train_data = data_df.iloc[train_data_start:train_data_end]
    valid_data = data_df[target_column].iloc[valid_data_start:valid_data_end]
    test_data = data_df.iloc[test_data_start:test_data_end]


    # valid_data のスライスを縦方向に結合
    new_valid_data = pd.DataFrame([])
    for i in range(0, n):
      sliced_data = valid_data.iloc[i : i+fl]
      new_valid_data = pd.concat([new_valid_data, sliced_data], axis=0)

    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    valid_tensor = torch.tensor(new_valid_data.values, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor)
    valid_dataset = TensorDataset(valid_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=fl, shuffle=False)

    return train_dataloader, valid_dataloader, train_tensor.shape, train_data, valid_data, test_data
  

  def dnpus_for_timesfm(self, df_normalized: pd.DataFrame, batch_size: int, input_patch_len: int = 32, output_patch_len: int = 128, fl: int = 128, ratio: float = 1.0):
    """
    - df_normalized: 正規化された時系列データ (N, 8) の DataFrame
    - batch_size: 1 バッチあたりのデータ数
    - input_patch_len: 1つの Patch の長さ（例: 32）
    - ratio: データの使用割合 (0.0 ~ 1.0)
    """
    n = int(len(df_normalized) * ratio // batch_size) - 1

    train_data_start = 0
    train_data_end = n*batch_size
    test_data_start = n*batch_size
    # test_data_end = n*batch_size + SL + fl + self.end_index

    # データの分割（時系列を維持）
    train_data = df_normalized.iloc[train_data_start:train_data_end]
    test_data = df_normalized.iloc[test_data_start:]

    # Input Data (queue: 0:6), Output Data (queue: 7)
    X_patches = []
    Y_patches = []

    for i in range(n - input_patch_len):
      X_patches.append(train_data[COLUMN_OBSERVABLE+COLUMN_TARGET].iloc[i:i+input_patch_len].values)   # (B, Col_inputs)
      Y_patches.append(train_data[COLUMN_TARGET].iloc[i+input_patch_len:i+input_patch_len+output_patch_len].values)  # (B, Col_outputs)

    X_tensor = torch.tensor(np.array(X_patches), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_patches), dtype=torch.float32).reshape(-1, output_patch_len, 1)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, train_data, test_data
  

  def dnpus_for_tcn(self, df: pd.DataFrame, sl: int, fl: int, batch_size: int, ratio: float):
    """
    Prepare the training, validation, and testing data loaders for DNPU task.
    
    Args:
      df (pd.DataFrame): The DataFrame containing the dataset with columns input_0 to input_6 and output_0
      fl (int): The number of steps to predict into the future
      batch_size (int): The batch size for training and testing data
    
    Returns:
      train_loader, valid_loader, test_loader: DataLoaders for training, validation, and testing data
    """
    # Extract features and targets (assuming the DataFrame is already normalized)
    df_train = df.head(int(ratio * len(df)))
    df_test = df.iloc[int(ratio * len(df)) : int(ratio * len(df)) + min(len(df), self.end_index)]

    features = df_train[COLUMN_INPUT].values
    targets = df_train[COLUMN_TARGET].values

    # Split the data into training (80%) and validation (20%)
    train_features, valid_features, train_targets, valid_targets = train_test_split(features, targets, test_size=0.2, shuffle=False)
    test_features, test_targets = df_test[COLUMN_INPUT].values, df_test[COLUMN_TARGET].values

    # Convert data to PyTorch tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    valid_features_tensor = torch.tensor(valid_features, dtype=torch.float32)
    valid_targets_tensor = torch.tensor(valid_targets, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

    # Create input-output pairs for sequence prediction
    def create_sequence_data(features, targets):
      inputs, outputs = [], []
      for i in range(sl, len(features) - fl):
        inputs.append(features[i-sl : i])
        # outputs.append(targets[i :i+fl])
        outputs.append(targets[i-1 :i])
      return torch.stack(inputs), torch.stack(outputs)

    # Prepare the training, validation, and testing sequences
    train_inputs, train_outputs = create_sequence_data(train_features_tensor, train_targets_tensor)
    valid_inputs, valid_outputs = create_sequence_data(valid_features_tensor, valid_targets_tensor)
    test_inputs, test_outputs = create_sequence_data(test_features_tensor, test_targets_tensor)
    print(train_targets.shape)
    print(valid_targets.shape)

    # Create DataLoader instances for batch processing
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    valid_dataset = torch.utils.data.TensorDataset(valid_inputs, valid_outputs)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_outputs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


  def dnpus_for_static(self, df: pd.DataFrame, sl: int, fl: int, batch_size: int, ratio: float):
    """
    Prepare the training, validation, and testing data loaders for DNPU task.
    
    Args:
      df (pd.DataFrame): The DataFrame containing the dataset with columns input_0 to input_6 and output_0
      fl (int): The number of steps to predict into the future
      batch_size (int): The batch size for training and testing data
    
    Returns:
      train_loader, valid_loader, test_loader: DataLoaders for training, validation, and testing data
    """
    # Extract features and targets (assuming the DataFrame is already normalized)
    df_train = df.head(int(ratio * len(df)))
    df_test = df.iloc[int(ratio * len(df)) : int(ratio * len(df)) + min(len(df), self.end_index)]

    features = df_train[COLUMN_INPUT].values
    targets = df_train[COLUMN_TARGET].values

    # Split the data into training (80%) and validation (20%)
    train_features, valid_features, train_targets, valid_targets = train_test_split(features, targets, test_size=0.2, shuffle=False)
    test_features, test_targets = df_test[COLUMN_INPUT].values, df_test[COLUMN_TARGET].values

    # Convert data to PyTorch tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    valid_features_tensor = torch.tensor(valid_features, dtype=torch.float32)
    valid_targets_tensor = torch.tensor(valid_targets, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

    # Create input-output pairs for sequence prediction
    def create_sequence_data(features, targets):
      inputs, outputs = [], []
      for i in range(sl, len(features) - fl):
        inputs.append(features[i-sl : i])
        outputs.append(targets[i :i+fl])
      return torch.stack(inputs), torch.stack(outputs)

    # Prepare the training, validation, and testing sequences
    train_inputs, train_outputs = create_sequence_data(train_features_tensor, train_targets_tensor)
    valid_inputs, valid_outputs = create_sequence_data(valid_features_tensor, valid_targets_tensor)
    test_inputs, test_outputs = create_sequence_data(test_features_tensor, test_targets_tensor)
    print(train_targets.shape)
    print(valid_targets.shape)

    # Create DataLoader instances for batch processing
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    valid_dataset = torch.utils.data.TensorDataset(valid_inputs, valid_outputs)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_outputs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader