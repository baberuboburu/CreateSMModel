from src.architectures.base import BaseArchitecture
import torch
import os


class Informer(BaseArchitecture):
  def __init__(self):
    super().__init__()


  def train(self):
    return

  def valid(self):
    return

  def test(self):
    return
  

  def load(self, model_dir: str, model_name: str, device: str):
    model_file_path = os.path.join(model_dir, f'{model_name}.pt')
    self.model.load_state_dict(torch.load(model_file_path, map_location=device, weights_only=True))
    self.model.to(device)
    return self.model
  