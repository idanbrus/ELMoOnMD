import abc
from abc import ABC
import pandas as pd

class Loader(ABC):

    @abc.abstractmethod
    def load_data(self):
        pass