import typing as tp

import numpy as np
import tensorflow as tf

from rlearn.config import TrainConfig
from rlearn.trainer.base import BaseTrainer


def set_config_to_trainer(
        config: TrainConfig,
        trainer: BaseTrainer,
):
    trainer.set_params(
        batch_size=config.batch_size,
        gamma=config.gamma,
        replace_ratio=config.replace_ratio,
        replace_step=config.replace_step,
        min_epsilon=config.min_epsilon,
        epsilon_decay=config.epsilon_decay,
    )
    trainer.build_model_from_config(config=config)
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
        learning_rates: tp.Sequence[float],
        log_dir: tp.Optional[str] = None,
        seed: tp.Optional[int] = None
) -> BaseTrainer:
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    if len(__TRAINER_MAP) == 0:
        _set_trainer_map(BaseTrainer, __TRAINER_MAP)
    trainer = __TRAINER_MAP[name](learning_rates=learning_rates, log_dir=log_dir)
    return trainer


def get_all():
    if len(__TRAINER_MAP) == 0:
        _set_trainer_map(BaseTrainer, __TRAINER_MAP)
    return __TRAINER_MAP
