from config.config import *
import torch
from tqdm import tqdm
import numpy as np



class Calculate():
  def __init__(self):
    pass
  

  def calculate_predictions(self, trainer, dataset, start=0, end=200, amplification=1.0):
    """
    Calculate predictions and true values for a given dataset within a specified range.
    
    Args:
    trainer (Trainer): The Hugging Face Trainer instance with the trained model.
    dataset (ForecastDFDataset): The ForecastDFDataset containing the time series data.
    start (int): The starting index for calculating predictions. Default is 0.
    end (int): The ending index for calculating predictions. Default is 200.
    amplification (float): Amplification factor applied to the data. Default is 1.0.
    
    Returns:
    (np.array, np.array): Tuple of true values and predicted values.
    """

    # Check if GPU is available and set the device accordingly (Only for Apple Silicon)
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(BACKEND)
    print(f"Using device: {device}")
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
      trainer.model.eval()
      trainer.model.to(device)
      
      for i in tqdm(range(start, min(end, len(dataset)))):
        batch = dataset[i]
        inputs = batch["past_values"].unsqueeze(0).to(device)
        targets = batch["future_values"].unsqueeze(0).to(device)
        # print(inputs.shape)
        # print(targets.shape)
        
        # Get the model's predictions
        output = trainer.model(inputs)
        predictions = output.prediction_outputs  # Assuming the predictions are in prediction_outputs attribute
        
        all_targets.append(amplification * targets.squeeze(0))
        all_predictions.append(amplification * predictions.squeeze(0))
        # break
  
    # all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    # all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_targets, all_predictions