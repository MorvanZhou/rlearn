import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.trainer.base import BaseTrainer


def set_config_to_trainer(
        config: TrainConfig,
        trainer: BaseTrainer,
):
    trainer.set_params(
        learning_rate=config.learning_rates,
        batch_size=config.batch_size,
        gamma=config.gamma,
        replace_ratio=config.replace_ratio,
        replace_step=config.replace_step,
        min_epsilon=config.min_epsilon,
        epsilon_decay=config.epsilon_decay,
    )
    trainer.set_model_encoder_from_config(config=config)
    trainer.set_replay_buffer(
        max_size=config.replay_buffer.max_size,
        buf=config.replay_buffer.buf,
    )


__TRAINER_MAP: tp.Dict[str, tp.Type[BaseTrainer]] = {}
__BASE_MODULE = BaseTrainer.__module__


def _set_trainer_map(cls, m: dict):
    for subclass in cls.__subclasses__():
        if subclass.__module__ != __BASE_MODULE and not subclass.__name__.startswith("_"):
            m[subclass.__name__] = subclass
        _set_trainer_map(subclass, m)


def get_trainer_by_name(
        name: str,
        log_dir: tp.Optional[str] = None,
        seed: tp.Optional[int] = None
) -> BaseTrainer:
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    if len(__TRAINER_MAP) == 0:
        _set_trainer_map(BaseTrainer, __TRAINER_MAP)
    trainer = __TRAINER_MAP[name](log_dir=log_dir)
    return trainer


def get_all():
    if len(__TRAINER_MAP) == 0:
        _set_trainer_map(BaseTrainer, __TRAINER_MAP)
    return __TRAINER_MAP


LearningRate = tp.TypeVar("LearningRate", tp.Sequence[float], float)


def parse_2_learning_rate(learning_rate: LearningRate) -> tp.Tuple[float, float]:
    if isinstance(learning_rate, (tuple, list)) and len(learning_rate) <= 2:
        l_len = len(learning_rate)
        if l_len == 1:
            l1, l2 = learning_rate[0]
        elif l_len == 2:
            l1, l2 = learning_rate[0], learning_rate[1]
        else:
            raise ValueError("the sequence length of the learning rate must greater than 1")
    else:
        l1, l2 = learning_rate, learning_rate
    return l1, l2


BatchReward = tp.TypeVar('BatchReward', tp.List[float], np.ndarray)


def general_average_estimation(
        value_model: keras.Model,
        batch_s: np.ndarray,
        batch_r: BatchReward,
        batch_done: tp.List[bool],
        s_: np.ndarray,
        gamma: float = 0.9,
        lam: float = 0.9,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    adv = []
    next_v = value_model.predict(
        np.expand_dims(np.array(s_, dtype=np.float32), axis=0), verbose=0).ravel()[0]
    vs = value_model.predict(batch_s, verbose=0).ravel()
    gae_lam = 0
    for i in range(len(batch_s) - 1, -1, -1):  # backward count
        non_terminate = 0. if batch_done[i] else 1.
        delta = batch_r[i] + gamma * next_v * non_terminate - vs[i]
        gae_lam = delta + gamma * lam * gae_lam * non_terminate
        adv.insert(0, gae_lam)
        next_v = vs[i]

    adv = np.array(adv, dtype=np.float32)
    returns = adv + vs
    adv = (adv - adv.mean()) / (adv.std() + 1e-4)
    return returns, adv


def discounted_reward(
        value_model: keras.Model,
        batch_s: np.ndarray,
        batch_r: BatchReward,
        batch_done: tp.List[bool],
        s_: np.ndarray,
        gamma: float = 0.9,
) -> np.ndarray:
    discounted_r = []
    vs_ = value_model.predict(
        np.expand_dims(np.array(s_, dtype=np.float32), axis=0),
        verbose=0).ravel()[0]
    for i in range(len(batch_s) - 1, -1, -1):  # backward count
        if batch_done[i]:
            vs_ = 0
        vs_ = batch_r[i] + gamma * vs_
        discounted_r.insert(0, vs_)
    returns = np.array(discounted_r, dtype=np.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns


def discounted_adv(
        value_model: keras.Model,
        batch_s: np.ndarray,
        reward: np.ndarray,
) -> np.ndarray:
    adv = reward - value_model.predict(batch_s, verbose=0).ravel()
    adv = (adv - adv.mean()) / (adv.std() + 1e-4)
    return adv
