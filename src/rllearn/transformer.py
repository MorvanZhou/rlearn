import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, a: tp.Union[float, int, tf.Tensor, np.ndarray]) -> tp.Union[float, tf.Tensor, np.ndarray]:
        pass


class ContinuousAction(BaseTransformer):
    def __init__(self, action_map: tp.List[tp.List[float]]):
        bound_width = []
        bound_min = []
        for i in range(len(action_map)):
            bound = action_map[i]
            bound_width.append(bound[1] - bound[0])
            bound_min.append(bound[0])
        self.bound_width_tf = tf.constant(bound_width, dtype=tf.float32)
        self.bound_min_tf = tf.constant(bound_min, dtype=tf.float32)
        self.bound_width_np = np.array(bound_width, dtype=np.float32)
        self.bound_min_np = np.array(bound_min, dtype=np.float32)

    def transform(self, a: tp.Union[float, int, tf.Tensor, np.ndarray]) -> tp.Union[float, tf.Tensor, np.ndarray]:
        if isinstance(a, (np.ndarray, float, int, np.floating, np.integer)):
            bound_width = self.bound_width_np
            bound_min = self.bound_min_np
            clip = np.clip
        elif isinstance(a, tf.Tensor):
            bound_width = self.bound_width_tf
            bound_min = self.bound_min_tf
            clip = tf.clip_by_value
        else:
            raise TypeError(f"action type should be tensor or np array, but get {type(a)}")
        clipped_a = clip(a, -1, 1) / 2 + 0.5
        scale_a = clipped_a * bound_width
        return scale_a + bound_min


class DiscreteAction(BaseTransformer):
    def __init__(self, action_map: tp.List[float]):
        self.action_map = action_map

    def transform(self, a: tp.Union[float, int, tf.Tensor, np.ndarray]) -> tp.Union[float, tf.Tensor, np.ndarray]:
        if isinstance(a, (np.ndarray, float, int, np.floating, np.integer)):
            index = int(a)
        elif isinstance(a, tf.Tensor):
            index = int(a.numpy())
        else:
            raise TypeError(f"action type should be tensor or np array, but get {type(a)}")
        return self.action_map[index]
