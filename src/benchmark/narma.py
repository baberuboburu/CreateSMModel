import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config.narma import *


class NARMA():
  def __init__(self):
    # NARMA2 Params
    self.alpha2 = ALPHA2
    self.beta2 = BETA2
    self.gamma2 = GAMMA2
    self.delta2 = DELTA2

    # NARMA10 Params
    self.alpha10 = ALPHA10
    self.beta10 = BETA10
    self.gamma10 = GAMMA10
    self.delta10 = DELTA10

    # Common
    self.sigma = SIGMA
    self.training_ratio = TRAINING_RATIO


  def create_input(self, T: int, seed: int = None):
    """
    Create input data as a pandas DataFrame.

    Args:
      T (int): Length of the input data.
      seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
      pd.DataFrame: Input data with one column 'u_k'.
    """
    if seed is not None:
      np.random.seed(seed)
    u_k = np.random.uniform(-self.sigma, self.sigma, T)

    return pd.DataFrame({'u_k': u_k})


  def create_target_narma2(self, input_data: pd.DataFrame):
    """
    Create target data as a pandas DataFrame based on the NARMA10 model.

    Args:
      input_data (pd.DataFrame): Input data with one column 'u_k'.

    Returns:
      pd.DataFrame: Target data with one column 'y_k'.
    """
    u_k = input_data['u_k'].values
    y_k = np.zeros_like(u_k)
    
    # Initialize first 2 values to 0
    for i in range(2):
      y_k[i] = 0

    for k in range(2, len(u_k)):
      y_k[k] = (
      self.alpha2 * y_k[k - 1]
      + self.beta2 * y_k[k - 1] * y_k[k - 2]
      + self.gamma2 * u_k[k] * u_k[k - 1]
      + self.delta2
    )

    return pd.DataFrame({'y_k': y_k})


  def create_target_narma10(self, input_data: pd.DataFrame):
    """
    Create target data as a pandas DataFrame based on the NARMA10 model.

    Args:
      input_data (pd.DataFrame): Input data with one column 'u_k'.

    Returns:
      pd.DataFrame: Target data with one column 'y_k'.
    """
    u_k = input_data['u_k'].values
    y_k = np.zeros_like(u_k)
    
    # Initialize first 10 values to 0
    for i in range(10):
      y_k[i] = 0

    for k in range(10, len(u_k)):
      y_k[k] = (
        self.alpha10 * y_k[k - 1]
        + self.beta10 * y_k[k - 1] * np.sum(y_k[k - 10:k])
        + self.gamma10 * u_k[k] * u_k[k - 9]
        + self.delta10
      )

    return pd.DataFrame({'y_k': y_k})
  

  def prepare_dataset(self, narma_input: pd.DataFrame, narma_target: pd.DataFrame):
    """
    Prepare the dataset by splitting it into training and testing datasets.

    Args:
      narma_input (pd.DataFrame): Input data as a pandas DataFrame.
      narma_target (pd.DataFrame): Target data as a pandas DataFrame.

    Returns:
      tuple: A tuple containing training data (narma_data_train) and testing data (narma_data_test).
    """
    # Combine input and target data along columns
    narma_data = pd.concat([narma_input, narma_target], axis=1)

    # Determine the split index for training and testing
    split_index = int(len(narma_data) * self.training_ratio)

    # Split the data into training and testing sets
    narma_data_train = narma_data.iloc[:split_index]
    narma_data_test = narma_data.iloc[split_index:]

    # narma_data = torch.utils.data.ConcatDataset([narma_data_train, narma_data_test])

    return narma_data, narma_data_train, narma_data_test


  def NRMSE(self, predicted: pd.DataFrame, target: pd.DataFrame):
    """
    Evaluate the NRMSE between predicted and target data.

    Args:
      predicted (pd.DataFrame): Predicted data with one column 'y_k'.
      target (pd.DataFrame): Target data with one column 'y_k'.

    Returns:
      float: Normalized Root Mean Square Error (NRMSE).
    """
    y_pred = predicted['y_k'].values
    y_target = target['y_hat'].values
    mse = np.mean((y_pred - y_target) ** 2)
    variance = np.var(y_target)

    return np.sqrt(mse / variance)
  

  def ridge_regression(self, predicted: np.array, target: np.array):
    """
    Perform Ridge regression to minimize the NRMSE error between y_pred and y_hat.

    Args:
      predicted (np.array): Predicted data as a numpy array.
      target (np.array): Target data as a numpy array.

    Returns:
      np.array: Adjusted y_pred after regression.
    """

    # Extract predicted and target values
    y_pred = torch.tensor(predicted, dtype=torch.float32).unsqueeze(1)
    y_target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    # Define Ridge regression model
    class RidgeRegression(nn.Module):
      def __init__(self, input_dim, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.alpha = alpha

      def forward(self, x):
        return self.linear(x)

      def l2_penalty(self):
        return self.alpha * torch.sum(self.linear.weight ** 2)

    model = RidgeRegression(input_dim=1, alpha=RIDGE_ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=RIDGE_LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(RIDGE_EPOCHS):  # Set a sufficient number of iterations
      optimizer.zero_grad()
      outputs = model(y_pred)
      loss = criterion(outputs, y_target) + model.l2_penalty()
      loss.backward()
      optimizer.step()

    # Predict adjusted values
    adjusted_y_pred = model(y_pred).detach().numpy()

    # Return as a DataFrame
    return adjusted_y_pred