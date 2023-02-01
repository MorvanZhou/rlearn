import typing as tp
from abc import ABC, abstractmethod

import numpy as np


class BaseReplayBuffer(ABC):
    name = __qualname__

    def __init__(self, max_size: int):
        self._max_size = max_size

        self._is_full = False
        self._is_empty = True

        self.data: tp.Dict[str, np.ndarray] = {}

    def put(self, **kwargs: tp.Union[int, float, np.ndarray]) -> None:
        return self.put_one(**kwargs)

    def preprocess_batch_data(self, **kwargs: np.ndarray) -> tp.Tuple[int, tp.Dict[str, np.ndarray]]:
        l_ = None
        for k in kwargs.keys():
            if not isinstance(kwargs[k], np.ndarray):
                kwargs[k] = np.array([[kwargs[k]]], dtype=type(kwargs[k]))
            batch_size = kwargs[k].shape[0]
            if l_ is None:
                l_ = batch_size
            if batch_size != l_:
                raise ValueError(f"batch size of '{k}'={batch_size}  is not the same as others={l_}")

        for k, v in kwargs.items():
            # init memory
            if k not in self.data:
                self.data[k] = np.zeros((self.max_size, *v.shape[1:]), dtype=v.dtype)
            else:
                if self.data[k].shape[1:] != v.shape[1:]:
                    raise ValueError(f"data {k} has different shape as {self.data[k].shape}, {v.shape}")
        if l_ is None:
            raise ValueError("no data is put in replay buffer")
        return l_, kwargs

    def put_one(self, **kwargs: tp.Union[int, float, np.ndarray]):
        for k in kwargs.keys():
            if not isinstance(kwargs[k], np.ndarray):
                kwargs[k] = np.array([kwargs[k]], dtype=type(kwargs[k]))
            else:
                if kwargs[k].ndim < 2:
                    kwargs[k] = np.expand_dims(kwargs[k], axis=0)
        self.put_batch(**kwargs)

    def is_empty(self):
        return self._is_empty

    def is_full(self):
        return self._is_full

    def get_current_loading(self) -> tp.Dict[str, np.ndarray]:
        if self._is_empty:
            raise ValueError("no data in replay buffer")
        data = {}
        p = self.current_loading_point
        for k, v in self.data.items():
            data[k] = v[:p]
        return data

    @property
    @abstractmethod
    def current_loading_point(self):
        pass

    @abstractmethod
    def try_weighting_loss(self, target, evaluated):
        pass

    @abstractmethod
    def put_batch(self, **kwargs: np.ndarray) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> tp.Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @property
    def max_size(self):
        return self._max_size
