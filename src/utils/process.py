import numpy as np
import pandas as pd
import torch
from tsfm.tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor


class Process():
  def __init__(self):
    pass
  

  def get_sampling_data(self, filename: str, activation_electrode_no: int, readout_electrode_no: int):
    """
    Read and process a .dat file to extract input and output data.

    Args:
      filename (str): Path to the .dat file.
      activation_electrode_no (int): Number of input electrodes.
      readout_electrode_no (int): Number of output electrodes.

    Returns:
      tuple: Inputs and outputs as NumPy arrays.
    """
    # Read the .dat file, skipping the header row that starts with '#'
    # df = pd.read_csv(filename)  # Simple csv file
    # numeric_data = df.iloc[:, 1:]
    df = pd.read_csv(filename, sep='\\s+', comment='#', header=None)  # IO.dat

    # Separate inputs and outputs
    inputs = df.iloc[:, :activation_electrode_no].values
    outputs = df.iloc[:, -readout_electrode_no:].values

    return inputs, outputs


  def load_and_process_data(self, data_path: str, activation_electrode_no: int, readout_electrode_no: int):
    # Load the raw data
    inputs, outputs = self.get_sampling_data(data_path, activation_electrode_no, readout_electrode_no)
    
    # Convert to pandas DataFrame
    df_inputs = pd.DataFrame(inputs, columns=[f'input_{i}' for i in range(activation_electrode_no)])
    df_outputs = pd.DataFrame(outputs, columns=[f'output_{i}' for i in range(readout_electrode_no)])
    
    # Combine inputs and outputs into one DataFrame
    df = pd.concat([df_inputs, df_outputs], axis=1)
    
    # Add additional columns required by TimeSeriesDataSet
    df['time_idx'] = np.arange(len(df))  # Assuming the data is sequential
    df['group_id'] = 0                   # Assuming a single time series for simplicity
    
    return df


  def prepare_learning_dataset(
    self, 
    df_normalized: pd.DataFrame, 
    rate_training_dataset: float, 
    rate_validation_dataset: float, 
    context_length=None, 
    forecast_length=None,
    **column_specifiers
    ):
    """
    Prepare learning datasets for either general time series or NARMA-specific tasks.

    Args:
      df_normalized (pd.DataFrame): The normalized input dataframe.
      rate_training_dataset (float): Ratio of training data.
      rate_validation_dataset (float): Ratio of validation data.
      context_length (int): Context length for the dataset.
      forecast_length (int): Forecast length for the dataset.
      **column_specifiers: Flexible keyword arguments for column specifications, such as timestamp_column, id_columns, target_columns, etc.

    Returns:
      tuple: full_dataset, train_dataset, valid_dataset, test_dataset
    """

    # Step 1: Split the dataset into train, valid, and test
    end_train = int(df_normalized.shape[0] * rate_training_dataset)
    end_valid = int(df_normalized.shape[0] * (rate_training_dataset + rate_validation_dataset))
    split_config = {
      "train": [0, end_train],
      "valid": [end_train, end_valid],
      "test": [end_valid, df_normalized.shape[0]],
    }

    # Step 2: Validate specified columns
    for key, columns in column_specifiers.items():
      if isinstance(columns, list):
        for col in columns:
          assert col in df_normalized.columns, f"Column {col} in {key} is missing"
      elif isinstance(columns, str):
        assert columns in df_normalized.columns, f"Column {columns} in {key} is missing"

    # Step 3: Initialize the TimeSeriesPreprocessor
    tsp = TimeSeriesPreprocessor(
      **column_specifiers,
      context_length=context_length,
      prediction_length=int(forecast_length),
      scaling=True,
      encode_categorical=False,
      scaler_type="minmax",
    )

    # Step 4: Create train, valid, and test datasets
    train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
      df_normalized, split_config
    )

    # Step 5: Concatenate all datasets into a full dataset
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset, test_dataset])

    return full_dataset, train_dataset, valid_dataset, test_dataset