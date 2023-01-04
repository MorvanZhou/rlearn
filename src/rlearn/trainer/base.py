import inspect
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tensorflow import keras

from rlearn import replaybuf, board
from rlearn.config import TrainConfig
from rlearn.replaybuf.base import BaseReplayBuffer


class BaseTrainer(ABC):
    is_on_policy = False

    def __init__(
            self,
            learning_rates: tp.Sequence[float],
            log_dir: tp.Optional[str] = None,
    ):
        super().__init__()
        if not isinstance(learning_rates, (tuple, list)):
            raise TypeError("learning rates must be tuple or list")

        self.model = None
        self.learning_rates: tp.Sequence[float] = learning_rates
        self.batch_size: int = 32

        self.replay_buffer: tp.Optional[BaseReplayBuffer] = None

        self.replace_ratio = 1.
        self.replace_step = 0
        self.min_epsilon = 0.
        self.epsilon_decay = 1e-3
        self.epsilon = 1.
        self.gamma = 0.9

        self._replace_counter = 0
        self.log_dir = log_dir
        self.board = None

    @abstractmethod
    def train_batch(self):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model_weights(self, path: str):
        pass

    @abstractmethod
    def predict(self, s: np.ndarray):
        pass

    @abstractmethod
    def store_transition(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_model_from_config(self, config: TrainConfig):
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    def trace(self, data: dict, step: int):
        self.board.trace(data, step=step)

    def set_replay_buffer(
            self,
            max_size: int = 1000,
            buf: tp.Union[str, tp.Type[replaybuf.base.BaseReplayBuffer]] = "RandomReplayBuffer"):
        if inspect.isclass(buf):
            name = buf.name
        else:
            name = buf
        self.replay_buffer = replaybuf.get_buffer_by_name(name=name, max_size=max_size)

    def _set_tensorboard(self, models):
        if self.log_dir is not None and self.board is None:
            board_dir = Path(self.log_dir) / "board"
            self.board = board.Tensorboard(
                logdir=str(board_dir), models=models, trace_weights=True)

    def set_params(
            self,
            batch_size: int = 32,
            gamma: float = 0.9,
            replace_ratio: float = 1.,
            replace_step: int = 0,
            min_epsilon: float = 0.1,
            epsilon_decay: float = 1e-3,
    ):
        self.batch_size = batch_size
        self.replace_ratio = replace_ratio
        self.replace_step = replace_step
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def replace_target_net(
            self,
            src: keras.Model,
            target: keras.Model,
            ratio: tp.Optional[float] = None
    ):
        """
        替代 target 网络。

        Args:
            src: 替代者网络
            target: 被替代网络
            ratio: 被替代网络，有多少被替代的幅度。例如 ratio=0.1, 表示原始 target 的占比是 0.9，
                src 的占比是 0.1，新 target 的更像原始 target
        """
        if ratio is None:
            ratio = self.replace_ratio
        if ratio > 1. or ratio < 0.:
            raise ValueError(f"replace ratio must in range of [0, 1], but get {ratio}")
        for layer_, layer in zip(target.layers, src.layers):
            for weights_, weights in zip(layer_.weights, layer.weights):
                if ratio == 1.:
                    weights_.assign(weights)
                else:
                    weights_.assign(weights_ * (1 - ratio) + weights * ratio)

    def try_replace_params(self, src, target, ratio: tp.Optional[float] = None) -> bool:
        """
        尝试替代 target 网络，如果 replace_step <= 0 或者 到了 replace_step 时，进行替换 target 网络的工作。

        Args:
            src: 替代者网络
            target: 被替代网络
            ratio: 被替代网络，有多少被替代的幅度。例如 ratio=0.1, 表示原始 target 的占比是 0.9，
                src 的占比是 0.1，新 target 的更像原始 target

        Return:
            bool: 是否执行了替换操作
        """
        replaced = False
        self._replace_counter += 1
        if self.replace_step <= 0 or self._replace_counter % self.replace_step == 0:
            self.replace_target_net(src, target, ratio)
            self._replace_counter = 0
            replaced = True
        return replaced

    def decay_epsilon(self):
        if self.epsilon <= self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon *= (1 - self.epsilon_decay)

    def try_process_replay_buffer(self):
        return
