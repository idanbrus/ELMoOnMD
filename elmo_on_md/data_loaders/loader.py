import abc
from abc import ABC

class Loader(ABC):

    @abc.abstractmethod
    def load_data(self):
        pass