import typing as tp

import numpy as np

from rlearn.replaybuf.base import BaseReplayBuffer


class RandomReplayBuffer(BaseReplayBuffer):
    name = __qualname__

    def __init__(self, max_size):
        super().__init__(max_size=max_size)

        self.__pointer = 0
        self.__all_indices = None

    def sample(self, batch_size: int) -> tp.Tuple:
        if self.pointer == 0 and not self.is_full():
            raise ValueError("replay buffer is empty")
        if self.__all_indices is None:
            self.__all_indices = np.arange(self.max_size)

        if self.is_full():
            indices = np.random.choice(self.__all_indices, size=batch_size, replace=False)
        else:
            if self.__pointer < batch_size:
                replace = True
            else:
                replace = False
            indices = np.random.choice(self.__all_indices[:self.__pointer], size=batch_size, replace=replace)

        ba = self.a[indices]
        br = self.r[indices]
        bs = self.s[indices]

        # next state
        if self.has_next_state:
            bs_ = bs[:, 1]
            bs = bs[:, 0]
            batch = (bs, ba, br, bs_)
        else:
            # no next state
            batch = (bs, ba, br, None)
        return batch

    def put_batch(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            s_: tp.Optional[np.ndarray] = None
    ):
        states, a, r = self.preprocess_batch_data(s, a, r, s_)

        batch_size = a.shape[0]
        start = self.__pointer
        end = self.__pointer + batch_size

        if self._is_empty and batch_size > 0:
            self._is_empty = False

        restart = False
        if end >= self.max_size:
            sub_batch_size = self.max_size - start
            self.s[start:, :] = states[-sub_batch_size:]
            self.a[start:] = a[-sub_batch_size:]
            self.r[start:] = r[-sub_batch_size:]
            start = 0
            end = (batch_size - sub_batch_size) % self.max_size
            batch_size = end
            restart = True

        self.s[start:end, :] = states[:batch_size]
        self.a[start:end] = a[:batch_size]
        self.r[start:end] = r[:batch_size]

        if restart:
            self.__pointer = batch_size
            self._is_full = True
        else:
            self.__pointer += batch_size

    def clear(self):
        self.__pointer = 0
        self._is_full = False
        self._is_empty = True

    @property
    def current_loading_point(self):
        if self.is_full():
            return self.max_size
        return self.pointer

    @property
    def pointer(self):
        return self.__pointer
