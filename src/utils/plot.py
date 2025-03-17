import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import stft
from typing import List
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Plot():
  def __init__(self):
    self.sampling_rate = 2000


  # Visualize the input and output series
  def plot_time_series(self, df: pd.DataFrame, activation_electrode_no: int, readout_electrode_no: int, plot_dir: str, plot_filename: str = 'time_series'):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot inputs
    for i in range(activation_electrode_no):
      axs[0].plot(df['time_idx'], df[f'input_{i}'], label=f'Series {i}')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Input Series')
    
    # Plot outputs
    for i in range(readout_electrode_no):
      axs[1].plot(df['time_idx'], df[f'output_{i}'], label=f'Output {i}', color='r')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Output Series')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


  def plot_original_time_series(self, df: pd.DataFrame, plot_dir: str, plot_filename: str = 'original_time_series'):
    plt.figure(figsize=(14, 7))
    plt.plot(df['output_0'], label='Time Series Data')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Output_0')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


  def plot_decompose_time_series(self, df: pd.DataFrame, plot_dir: str):
    result = seasonal_decompose(df['output_0'], model='additive', period=365)
    result.plot()
    plot_filename = 'seasonal_decompose'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()
    
    # Trend Component
    trend = result.trend.dropna()
    plt.figure(figsize=(14, 7))
    plt.plot(trend, label='Trend')
    plt.title('Trend Component')
    plt.xlabel('Date')
    plt.ylabel('Output_0')
    plt.legend()
    plot_filename = 'trend_component'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()

    # Seasonal Component
    seasonal = result.seasonal.dropna()
    plt.figure(figsize=(14, 7))
    plt.plot(seasonal, label='Seasonality')
    plt.title('Seasonal Component')
    plt.xlabel('Date')
    plt.ylabel('Output_0')
    plt.legend()
    plot_filename = 'seasonal_component'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


    # Residual Component
    residual = result.resid.dropna()
    plt.figure(figsize=(14, 7))
    plt.plot(residual, label='Residuals')
    plt.title('Residual Component')
    plt.xlabel('Date')
    plt.ylabel('Output_0')
    plt.legend()
    plot_filename = 'residual_component'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


  def plot_stationarity(self, df: pd.DataFrame):
    adf_result = adfuller(df['output_0'])
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    for key, value in adf_result[4].items():
      print(f'Critical Value {key}: {value}')


  def plot_acf_and_pacf(self, df: pd.DataFrame, plot_dir: str, plot_filename: str = 'acf_and_pacf'):
    plt.figure(figsize=(14, 7))
    
    # ACF Plot
    plt.subplot(211)
    plot_acf(df['output_0'], lags=50, ax=plt.gca())

    # PACF Plot
    plt.subplot(212)
    plot_pacf(df['output_0'], lags=50, ax=plt.gca())

    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()
  

  def plot_ground_truth(self, df: pd.DataFrame, start_idx: int, plot_length: int, plot_dir: str, plot_filename: str = 'ground_truth'):
    plt.figure(figsize=(14, 7))
    plt.plot(df['output_0'][start_idx:start_idx + plot_length], label='Ground Truth')
    plt.title('Ground Truth')
    plt.xlabel('Date')
    plt.ylabel('Output_0')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()
  

  def plot_predictions(self, true_values, predicted_values, plot_dir, plot_prefix="valid", channel=0, start=0, end=200):
    """
    Plot the true values and predicted values for a time series dataset within a specified range.
    
    Args:
    true_values (np.array): Array of true values.
    predicted_values (np.array): Array of predicted values.
    plot_dir (str): Directory to save the plot.
    plot_prefix (str): Prefix for the plot file name.
    channel (int): The channel to visualize. Default is 0.
    start (int): The starting index for plotting. Default is 0.
    end (int): The ending index for plotting. Default is 200.
    """
    # Ensure the end index does not exceed the length of the dataset
    end = min(end, len(true_values))
    
    # Check the dimensions and adjust if necessary
    if true_values.ndim == 1:
      y_true_flat = true_values[start:end]
      y_pred_flat = predicted_values[start:end]
    elif true_values.ndim == 2 and true_values.shape[1] > channel:
      y_true_flat = true_values[start:end, channel].flatten()
      y_pred_flat = predicted_values[start:end, channel].flatten()
    else:
      raise IndexError(f"Channel index {channel} is out of bounds for axis 1 with size {true_values.shape[1]}")
    
    # Create a single plot
    plt.figure(figsize=(14, 7))
    
    # Plot true values
    plt.plot(y_true_flat, label="True", linestyle="-", color="blue", linewidth=2)
    
    # Plot predicted values
    plt.plot(y_pred_flat, label="Predicted", linestyle="--", color="orange", linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Real vs Predicted Outputs')
    plt.legend()
    
    # Save the plot
    plot_filename = f"{plot_prefix}_pred_vs_true_ch_{str(channel)}_range_{start}_{end}.png"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


  def plot_forecast(self, train_tensor, test_tensor, forecast_tensor, sl, fl, plot_dir: str, plot_filename: str, channel_idx: int):
    history = train_tensor[channel_idx, -(sl-fl):].detach().numpy()
    true = test_tensor[channel_idx, 0:fl].detach().numpy()
    pred = forecast_tensor[channel_idx, 0:fl].detach().numpy()

    plt.figure(figsize=(14, 7))

    # Plotting the first time series from history
    plt.plot(range(len(history)), history, label='History (512 timesteps)', c='darkblue')

    # Plotting ground truth and prediction
    offset = len(history)
    plt.plot(range(offset, offset + len(true)), true, label='Ground Truth (128 timesteps)', color='darkblue', linestyle='--', alpha=0.5)
    plt.plot(range(offset, offset + len(pred)), pred, label='Forecast (128 timesteps)', color='red', linestyle='--')

    plt.title(f"'Real vs Predicted Outputs'", fontsize=18)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()


  def plot_training_loss(self, losses: List[float], plot_dir: str, epochs: int, d: float = 1e-5):
    # find stable epoch
    n = self._find_stable_epoch(losses, d)

    plt.figure(figsize=(14, 7))
    plt.plot(range(1, len(losses)), losses[1:], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.xticks(range(1, len(losses), 10))
    plt.axvline(x=n, color='red', linestyle='--', label=f'n={n}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'Training_Loss_{epochs}'))
    plt.show()


  def _find_stable_epoch(self, losses: List[float], d: float) -> int:
    for j in range(len(losses) - 1):
      if all(abs(losses[i] - losses[i + 1]) <= d for i in range(j, len(losses) - 1)):
        return j
    return len(losses) - 1


  def plot_fft(self, true_values, plot_dir, start=0, end=10000, prefix: str = ''):
    """
    Plot the FFT using target values within a specified range.
    
    Args:
    true_values (np.array): Array of true values.
    plot_dir (str): Directory to save the plot.
    start (int): The starting index for plotting. Default is 0.
    end (int): The ending index for plotting. Default is 20.
    """
    # Check if true_values is a valid array
    if not isinstance(true_values, np.ndarray):
      raise TypeError("true_values must be a NumPy array.")
    
    # Ensure valid range
    num_values = len(true_values)
    if num_values == 0:
      raise ValueError("true_values is empty.")

    if start < 0:
      start = 0
    if end > num_values:
      end = num_values
    if start >= end:
      raise ValueError(f"Invalid range for FFT analysis: start={start}, end={end}, length={num_values}")

    # Extract the required range
    signal_segment = true_values[start:end] - np.mean(true_values[start:end])

    # Compute FFT
    N = len(signal_segment)
    fft_values = fft(signal_segment)

    freqs = fftfreq(N, d=1/self.sampling_rate)

    # Plot FFT magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:N//40], np.abs(fft_values[:N//40]))
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid()

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"fft_{prefix}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"FFT plot saved at: {plot_path}")


  def plot_stft(self, true_values, plot_dir, start=0, end=10000, prefix: str = ''):
    """
    Plot the STFT using target values within a specified range.
    
    Args:
    true_values (np.array): Array of true values.
    plot_dir (str): Directory to save the plot.
    start (int): The starting index for plotting. Default is 0.
    end (int): The ending index for plotting. Default is 20.
    """
    # Ensure valid range
    max_index = max(end, len(true_values))
    true_values = true_values[:max_index]
    signal_segment = true_values[start:end] - np.mean(true_values[start:end])
    
    # Compute STFT
    f, t, Zxx = stft(signal_segment, fs=self.sampling_rate, nperseg=128)
    
    # Plot STFT magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title("STFT Magnitude Spectrum")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Magnitude")
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"stft_{prefix}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"STFT plot saved at: {plot_path}")
