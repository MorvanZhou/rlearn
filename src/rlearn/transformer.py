import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseTransformer(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def transform(self, a: tp.Union[float, int, tf.Tensor, np.ndarray]) -> tp.Union[float, tf.Tensor, np.ndarray]:
        pass

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class ContinuousAction(BaseTransformer):
    def __init__(
            self,
            bound: tp.Union[tp.Sequence[tp.Sequence[float]], tp.Sequence[float]],
    ):
        super().__init__(params=bound)
        _bound_width = []
        _bound_min = []
        if not isinstance(bound[0], (tuple, list)):
            bound = [bound, ]

        bound: tp.Sequence[tp.Sequence[float]]

        for i in range(len(bound)):
            _bound = bound[i]
            _bound_width.append(_bound[1] - _bound[0])
            _bound_min.append(_bound[0])
        self.bound_width_tf = tf.constant(_bound_width, dtype=tf.float32)
        self.bound_min_tf = tf.constant(_bound_min, dtype=tf.float32)
        self.bound_width_np = np.array(_bound_width, dtype=np.float32)
        self.bound_min_np = np.array(_bound_min, dtype=np.float32)

    def transform(self, action: tp.Union[float, int, tf.Tensor, np.ndarray]) -> tp.Union[float, tf.Tensor, np.ndarray]:
        if isinstance(action, (np.ndarray, float, int, np.floating, np.integer)):
            bound_width = self.bound_width_np
            bound_min = self.bound_min_np
            clip = np.clip
        elif isinstance(action, tf.Tensor):
            bound_width = self.bound_width_tf
            bound_min = self.bound_min_tf
            clip = tf.clip_by_value
        else:
            raise TypeError(f"action type should be tensor or np array, but get {type(action)}")
        clipped_a = clip(action, -1, 1) / 2 + 0.5
        scale_a = clipped_a * bound_width
        return scale_a + bound_min


class DiscreteAction(BaseTransformer):
    def __init__(
            self,
            actions: tp.Sequence[tp.Union[float, str]],
    ):
        super().__init__(params=actions)
        self.actions = actions

    def transform(
            self,
            action: tp.Union[float, int, tf.Tensor, np.ndarray],
    ) -> tp.Union[float, str, tf.Tensor, np.ndarray]:
        if isinstance(action, (np.ndarray, float, int, np.floating, np.integer)):
            index = int(action)
        elif isinstance(action, tf.Tensor):
            index = int(action.numpy())
        else:
            raise TypeError(f"action type should be tensor or np array, but get {type(action)}")
        return self.actions[index]
