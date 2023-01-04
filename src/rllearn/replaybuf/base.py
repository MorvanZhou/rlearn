import typing as tp
from abc import ABC, abstractmethod

import numpy as np


class BaseReplayBuffer(ABC):
    name = __qualname__

    def __init__(self, max_size: int):
        self._max_size = max_size
        self.has_next_state = False

        self._is_full = False
        self._is_empty = True

        self.s = None
        self.a = None
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)

    def put(
            self,
            s: np.ndarray,
            a: tp.Union[int, float, np.ndarray],
            r: float,
            s_: tp.Optional[np.ndarray] = None
    ) -> None:
        return self.put_one(s, a, r, s_)

    def preprocess_batch_data(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            s_: tp.Optional[np.ndarray] = None
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if r.ndim == 1:
            r = r[:, None]
        if r.ndim > 2:
            raise ValueError(f"wrong reward dimension='{r.ndim}', batch reward must be 2d array")
        # init state memory
        if self.s is None:
            if s_ is None:
                shape = (self.max_size, *s.shape[1:])
            else:
                self.has_next_state = True
                shape = (self.max_size, 2, *s.shape[1:])

            self.s = np.zeros(shape, dtype=np.float32)
        if self.a is None:
            self.a = np.zeros((self.max_size, *a.shape[1:]), dtype=np.float32)

        if self.has_next_state and s.shape != s_.shape:
            raise ValueError("shape mismatch: current state and next state")
        if not (a.shape[0] == s.shape[0] == r.shape[0]):
            raise ValueError("action/state/reward batch size not the same")
        if a.ndim < 2:
            raise ValueError("action must has at least 2 dimensions")

        if self.has_next_state:
            s = np.expand_dims(s, axis=1)
            s_ = np.expand_dims(s_, axis=1)
            states = np.concatenate([s, s_], axis=1)
        else:
            states = s
        return states, a, r

    def put_one(
            self,
            s: np.ndarray,
            a: tp.Union[int, float, np.ndarray],
            r: float,
            s_: tp.Optional[np.ndarray] = None
    ):
        if isinstance(a, (int, float, np.int8, np.int32, np.int64)) or not isinstance(a, np.ndarray):
            a = np.array([[a]], dtype=np.float32)
        if a.ndim < 2:
            a = np.expand_dims(a, axis=0)
        s = np.expand_dims(s, axis=0)
        s_ = np.expand_dims(s_, axis=0)
        r = np.array([[r]], dtype=np.float32)
        self.put_batch(s, a, r, s_)

    def process_state_data(self, s, s_):
        if self.has_next_state:
            s = np.expand_dims(s, axis=1)
            s_ = np.expand_dims(s_, axis=1)
            states = np.concatenate([s, s_], axis=1)
        else:
            states = s
        return states

    def empty(self):
        return self._is_empty

    def is_full(self):
        return self._is_full

    def get_current_loading(self) -> (np.ndarray, np.ndarray, np.ndarray):
        if self.s is None or self.a is None or self.r is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        all_s = self.s[:self.current_loading_point]
        a = self.a[:self.current_loading_point]
        r = self.r[:self.current_loading_point]
        return all_s, a, r

    @property
    @abstractmethod
    def current_loading_point(self):
        pass

    @abstractmethod
    def put_batch(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            s_: tp.Optional[np.ndarray] = None
    ) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @property
    def max_size(self):
        return self._max_size
