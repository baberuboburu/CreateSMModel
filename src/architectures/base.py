from src.utils.process import Process
from src.utils.normalize import Normalize
from src.utils.plot import Plot
from src.utils.prepare_data import PrepareData
from src.utils.calculate import Calculate


class BaseArchitecture():
  def __init__(self):
    self._process = Process()
    self._normalize = Normalize()
    self._plot = Plot()
    self._prepare_data = PrepareData()
    self._calculate = Calculate()
    
    self.start_index = 0
    self.end_index = 10000
