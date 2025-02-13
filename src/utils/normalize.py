import pandas as pd


class Normalize():
  def __init__(self):
    pass


  def min_max_normalize(self, df: pd.DataFrame, feature_columns, target_columns):
    """
    Apply min-max normalization to specified feature and target columns.
    X' = (X - X_min) / (X_max - X_min)
    
    Args:
      df (pd.DataFrame): Input DataFrame.
      feature_columns (list): List of feature column names to normalize.
      target_columns (list): List of target column names to normalize.

    Returns:
      pd.DataFrame: Normalized DataFrame.
    """
    df_normalized = df.copy()
    for column in feature_columns + target_columns:
      min_val = df[column].min()
      max_val = df[column].max()
      df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    return df_normalized


  def instance_normalize(self, df: pd.DataFrame, feature_columns, target_columns, epsilon=1e-8):
    """
    Apply instance normalization (row-wise normalization).
    X' = (X - mean_row) / (std_row + epsilon)

    Args:
      df (pd.DataFrame): Input DataFrame.
      feature_columns (list): List of feature column names to normalize.
      target_columns (list): List of target column names to normalize.
      epsilon (float): Small value to prevent division by zero.

    Returns:
      pd.DataFrame: Normalized DataFrame.
    """
    df_normalized = df.copy()
    for index, row in df.iterrows():
      row_features = row[feature_columns + target_columns]
      mean = row_features.mean()
      std = row_features.std()
      df_normalized.loc[index, feature_columns + target_columns] = (row_features - mean) / (std + epsilon)
    return df_normalized


  def batch_normalize(self, df: pd.DataFrame, feature_columns, target_columns, epsilon=1e-8):
    """
    Apply batch normalization (column-wise normalization).
    X' = (X - mean_batch) / (std_batch + epsilon)

    Args:
      df (pd.DataFrame): Input DataFrame.
      feature_columns (list): List of feature column names to normalize.
      target_columns (list): List of target column names to normalize.
      epsilon (float): Small value to prevent division by zero.

    Returns:
      pd.DataFrame: Normalized DataFrame.
    """
    df_normalized = df.copy()
    for column in feature_columns + target_columns:
      mean = df[column].mean()
      std = df[column].std()
      df_normalized[column] = (df[column] - mean) / (std + epsilon)
    return df_normalized