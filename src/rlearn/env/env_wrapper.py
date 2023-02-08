import typing as tp
from abc import ABC, abstractmethod

import numpy as np

State = tp.TypeVar("State", np.ndarray, tp.Dict[str, tp.Union[np.ndarray, float, bool, str, int]])


class EnvWrapper(ABC):
    @abstractmethod
    def reset(self) -> State:
        # return state
        pass

    @abstractmethod
    def step(self, a) -> tp.Tuple[State, float, bool]:
        # return (state, reward, done)
        pass

    def load(self, map_data: tp.Any):
        raise NotImplemented

    def render(self):
        raise NotImplemented
