import typing as tp

import numpy as np
import tensorflow as tf

from rlearn.replaybuf.base import BaseReplayBuffer


class RandomReplayBuffer(BaseReplayBuffer):
    name = __qualname__

    def __init__(self, max_size):
        super().__init__(max_size=max_size)

        self.__pointer = 0
        self.__all_indices = None

    def sample(self, batch_size: int) -> tp.Dict[str, np.ndarray]:
        if self.pointer == 0 and not self.is_full():
            raise ValueError("replay buffer is empty")
        if self.__all_indices is None:
            self.__all_indices = np.arange(self.max_size)

        if self.is_full():
            replace = batch_size > self.max_size
            indices = np.random.choice(self.__all_indices, size=batch_size, replace=replace)
        else:
            if self.__pointer < batch_size:
                replace = True
            else:
                replace = False
            indices = np.random.choice(self.__all_indices[:self.__pointer], size=batch_size, replace=replace)

        batch = {}
        for k, v in self.data.items():
            batch[k] = v[indices]
        return batch

    def put_batch(self, **kwargs: np.ndarray):
        batch_size, data = self.preprocess_batch_data(**kwargs)

        start = self.__pointer
        end = self.__pointer + batch_size

        if self._is_empty and batch_size > 0:
            self._is_empty = False

        restart = False
        if end >= self.max_size:
            sub_batch_size = self.max_size - start
            for k, v in data.items():
                self.data[k][start:] = v[-sub_batch_size:]
            start = 0
            end = (batch_size - sub_batch_size) % self.max_size
            batch_size = end
            restart = True

        for k, v in data.items():
            self.data[k][start:end] = v[:batch_size]

        if restart:
            self.__pointer = batch_size
            self._is_full = True
        else:
            self.__pointer += batch_size

    def clear(self):
        self.__pointer = 0
        self._is_full = False
        self._is_empty = True
        self.data.clear()

    def try_weighting_loss(self, target, evaluated):
        td = target - evaluated
        return tf.reduce_mean(tf.square(td))

    @property
    def current_loading_point(self):
        if self.is_full():
            return self.max_size
        return self.pointer

    @property
    def pointer(self):
        return self.__pointer
