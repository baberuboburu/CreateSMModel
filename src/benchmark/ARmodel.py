import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config.armodel import *


class ARModel():
  def __init__(self):
    # Auto Regressive Model (AR1)
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.delta = DELTA
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


  def create_target_ar1(self, input_data: pd.DataFrame):
    """
    Create target data as a pandas DataFrame based on the AR(1) model.

    Args:
      input_data (pd.DataFrame): Input data with one column 'u_k'.

    Returns:
      pd.DataFrame: Target data with one column 'y_k'.
    """
    u_k = input_data['u_k'].values
    y_k = np.zeros_like(u_k)
    
    # Initialize the first value to 0
    y_k[0] = 0

    for k in range(1, len(u_k)):
      y_k[k] = (
        self.alpha * y_k[k - 1]
        + self.gamma * u_k[k]
        + self.delta
      )

    return pd.DataFrame({'y_k': y_k})


  def prepare_dataset(self, ar1_input: pd.DataFrame, ar1_target: pd.DataFrame):
    """
    Prepare the dataset by splitting it into training and testing datasets.

    Args:
      ar1_input (pd.DataFrame): Input data as a pandas DataFrame.
      ar1_target (pd.DataFrame): Target data as a pandas DataFrame.

    Returns:
      tuple: A tuple containing training data (ar1_data_train) and testing data (ar1_data_test).
    """
    # Combine input and target data along columns
    ar1_data = pd.concat([ar1_input, ar1_target], axis=1)

    # Determine the split index for training and testing
    split_index = int(len(ar1_data) * self.training_ratio)

    # Split the data into training and testing sets
    ar1_data_train = ar1_data.iloc[:split_index]
    ar1_data_test = ar1_data.iloc[split_index:]

    # ar1_data = torch.utils.data.ConcatDataset([ar1_data_train, ar1_data_test])

    return ar1_data, ar1_data_train, ar1_data_test


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
