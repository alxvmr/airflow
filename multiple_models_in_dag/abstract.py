from abc import ABC
import pandas as pd
import numpy as np

class Model(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def name(self):
        raise NotImplementedError
    
class DataRepository(ABC):
    @abstractmethod
    def get_data(self, *name) -> dict:
        raise NotImplementedError