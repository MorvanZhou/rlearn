import typing as tp
from abc import ABC, abstractmethod

import numpy as np


class EnvWrapper(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, a) -> tp.Tuple[np.ndarray, float, bool]:
        # return (state, reward, done)
        pass

    def render(self):
        raise NotImplemented
