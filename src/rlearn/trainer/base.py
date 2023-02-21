import inspect
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras

import rlearn.type
from rlearn import replaybuf, board
from rlearn.config import TrainConfig
from rlearn.model.base import BaseRLModel
from rlearn.model.rnd import RND
from rlearn.replaybuf.base import BaseReplayBuffer
from rlearn.transformer import BaseTransformer


@dataclass
class TrainResult:
    value: tp.Dict[str, float]
    model_replaced: bool = False


class BaseTrainer(ABC):
    name = __qualname__

    def __init__(
            self,
            log_dir: tp.Optional[str] = None,
    ):
        super().__init__()

        self.model: tp.Optional[BaseRLModel] = None
        self.learning_rate: tp.Union[tp.Sequence[float], float] = 0.001
        self.batch_size: int = 32

        # default buffer
        self.replay_buffer: BaseReplayBuffer = replaybuf.RandomReplayBuffer(max_size=2000)

        self.replace_ratio = 1.
        self.replace_step = 0
        self.min_epsilon = 0.
        self.epsilon_decay = 1e-3
        self.epsilon = 1.
        self.gamma = 0.9
        self.rnd = None  # random network distillation

        self._replace_counter = 0
        self.log_dir = log_dir
        self.board = None

    @property
    def is_on_policy(self):
        return self.model.is_on_policy

    @abstractmethod
    def _set_default_optimizer(self):
        pass

    @abstractmethod
    def train_batch(self) -> TrainResult:
        pass

    @abstractmethod
    def predict(self, s: np.ndarray):
        pass

    @abstractmethod
    def store_transition(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_model_encoder_from_config(self, config: TrainConfig):
        pass

    @abstractmethod
    def set_model_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_gradients(self) -> tp.Tuple[TrainResult, tp.Optional[tp.Dict[str, tp.Dict[str, list]]]]:
        pass

    @abstractmethod
    def apply_flat_gradients(self, gradients: np.ndarray):
        pass

    def set_rl_model(self, model: BaseRLModel):
        if not isinstance(model, self.model.__class__):
            raise TypeError(f"model must be {self.model.__class__}, but got {type(model)}")
        self.model = model

    def compute_flat_gradients(self) -> tp.Optional[np.ndarray]:
        _, grads = self.compute_gradients()
        if grads is None:
            return grads
        flat_grads = []
        keys = list(grads.keys())
        keys.sort()
        for gd_key in keys:
            for g in grads[gd_key]["g"]:
                flat_grads.append(g.numpy().ravel())
        flat_grads = np.concatenate(flat_grads, dtype=np.float32)
        return flat_grads

    def save_model_weights(self, path: str):
        self.model.save_weights(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model.load(path)

    def trace(self, data: dict, step: int):
        if self.board is None:
            raise AttributeError("tensorboard is not defined")
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

    def _set_tensorboard(self, models: tp.List[keras.Model]):
        if self.log_dir is not None and self.board is None:
            board_dir = Path(self.log_dir) / "board"
            self.board = board.Tensorboard(
                logdir=str(board_dir), models=models, trace_weights=True)

    def set_params(
            self,
            learning_rate: tp.Union[tp.Sequence[float], float] = 1e-4,
            batch_size: int = 32,
            gamma: float = 0.9,
            replace_ratio: float = 1.,
            replace_step: int = 0,
            min_epsilon: float = 0.1,
            epsilon_decay: float = 1e-3,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replace_ratio = replace_ratio
        self.replace_step = replace_step
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def replace_target_net(
            self,
            source: tp.Union[tp.Sequence[keras.Model], keras.Model],
            target: tp.Union[tp.Sequence[keras.Model], keras.Model],
            ratio: tp.Optional[float] = None
    ):
        """
        替代 target 网络。

        Args:
            source: 替代者网络或网络列表
            target: 被替代网络或网络列表
            ratio: 被替代网络，有多少被替代的幅度。例如 ratio=0.1, 表示原始 target 的占比是 0.9，
                src 的占比是 0.1，新 target 的更像原始 target
        """
        if ratio is None:
            ratio = self.replace_ratio
        if ratio > 1. or ratio < 0.:
            raise ValueError(f"replace ratio must in range of [0, 1], but get {ratio}")
        if isinstance(source, (tuple, list)):
            src_seq = source
        else:
            src_seq = [source, ]
        if isinstance(target, (tuple, list)):
            target_seq = target
        else:
            target_seq = [target, ]

        for s, t in zip(src_seq, target_seq):
            for layer_, layer in zip(t.layers, s.layers):
                for weights_, weights in zip(layer_.weights, layer.weights):
                    if ratio == 1.:
                        weights_.assign(weights)
                    else:
                        weights_.assign(weights_ * (1 - ratio) + weights * ratio)

    def try_replace_params(
            self,
            source: tp.Union[tp.Sequence[keras.Model], keras.Model],
            target: tp.Union[tp.Sequence[keras.Model], keras.Model],
            ratio: tp.Optional[float] = None,
    ) -> bool:
        """
        尝试替代 target 网络，如果 replace_step <= 0 或者 到了 replace_step 时，进行替换 target 网络的工作。

        Args:
            source: 替代者网络或网络列表
            target: 被替代网络或网络列表
            ratio: 被替代网络，有多少被替代的幅度。例如 ratio=0.1, 表示原始 target 的占比是 0.9，
                src 的占比是 0.1，新 target 的更像原始 target

        Return:
            bool: 是否执行了替换操作
        """
        replaced = False
        self._replace_counter += 1
        if self.replace_step <= 0 or self._replace_counter % self.replace_step == 0:
            self.replace_target_net(source, target, ratio)
            self._replace_counter = 0
            replaced = True
        return replaced

    def decay_epsilon(self):
        if self.epsilon <= self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon *= (1 - self.epsilon_decay)

    def add_rnd(
            self,
            target: keras.Model,
            predictor: tp.Optional[keras.Model] = None,
            learning_rate: float = 1e-4,
    ):
        self.rnd = RND(target=target, predictor=predictor, learning_rate=learning_rate)

    def try_combine_int_ext_reward(self, ext_reward, next_state: np.ndarray):
        if self.rnd is None:
            return ext_reward
        scalar = False
        if isinstance(ext_reward, (int, float)):
            scalar = True
            next_state = next_state[None, :]
            ext_reward = np.array([ext_reward, ])
        elif isinstance(ext_reward, (list, tuple)):
            ext_reward = np.array(ext_reward)

        if ext_reward.ndim > 1:
            raise ValueError(f"ext_reward has dim={ext_reward.ndim}, but only accept 1")
        int_reward = self.rnd.intrinsic_reward(next_state)
        total_reward = ext_reward + int_reward
        if scalar:
            return total_reward[0]

        return total_reward

    def set_action_transformer(self, transformer: BaseTransformer):
        self.model.set_action_transformer(transformer)

    def map_action(self, action: rlearn.type.Action) -> rlearn.type.Action:
        return self.model.map_action(action)
