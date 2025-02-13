class EarlyStopping():
  def __init__(self, patience=5, min_delta=1e-6):
    self.patience = patience
    self.min_delta = min_delta
    self.best_loss = float('inf')
    self.counter = 0


  def __call__(self, val_loss):
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
    else:
      self.counter += 1
      print(f'Early stopping Counter is -- {self.counter} --')

    if self.counter >= self.patience:
      print("Early stopping triggered!")
      return True
    return False