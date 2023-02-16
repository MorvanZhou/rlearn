import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

import rlearn
from rlearn.config import TrainConfig
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer


def set_trainer_action_transformer(trainer: BaseTrainer, action_transform_config: list):
    if trainer.model.is_discrete_action:
        at = rlearn.transformer.DiscreteAction(action_transform_config)
    else:
        at = rlearn.transformer.ContinuousAction(action_transform_config)
    trainer.set_action_transformer(at)


def set_config_to_trainer(
        config: TrainConfig,
        trainer: BaseTrainer,
) -> None:
    """
    Set training parameters, model structure, action transformer, random network distillation, replay buffer

    Args:
        config (TrainConfig): training config
        trainer (BaseTrainer): trainer object
    """
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

    if config.action_transform is not None and len(config.action_transform) > 0:
        set_trainer_action_transformer(trainer=trainer, action_transform_config=config.action_transform)

    if config.random_network_distillation is not None:
        if config.random_network_distillation.predictor is None:
            predictor_conf = config.random_network_distillation.target
        else:
            predictor_conf = config.random_network_distillation.predictor
        predictor = build_encoder_from_config(predictor_conf, trainable=True)
        target = build_encoder_from_config(config.random_network_distillation.target)
        trainer.add_rnd(
            target=target, predictor=predictor, learning_rate=config.random_network_distillation.learning_rate)

    trainer.set_replay_buffer(
        max_size=config.replay_buffer.max_size,
        buf=config.replay_buffer.buf,
    )


__TRAINER_MAP: tp.Dict[str, tp.Type[BaseTrainer]] = {}
__BASE_MODULE = BaseTrainer.__module__


def _set_trainer_map(cls, m: dict):
    for subclass in cls.__subclasses__():
        if subclass.__module__ != __BASE_MODULE \
                and not subclass.__name__.startswith("_") \
                and subclass.__module__.startswith(rlearn.trainer.__name__):
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


def get_all() -> tp.Dict[str, tp.Type[BaseTrainer]]:
    if len(__TRAINER_MAP) == 0:
        _set_trainer_map(BaseTrainer, __TRAINER_MAP)
    return __TRAINER_MAP


def parse_2_learning_rate(learning_rate: rlearn.type.LearningRate) -> tp.Tuple[float, float]:
    if isinstance(learning_rate, (tuple, list)) and len(learning_rate) <= 2:
        l_len = len(learning_rate)
        if l_len == 1:
            l1 = l2 = learning_rate[0]
        elif l_len == 2:
            l1, l2 = learning_rate[0], learning_rate[1]
        else:
            raise ValueError("the sequence length of the learning rate must greater than 1")
    else:
        l1, l2 = learning_rate, learning_rate
    return l1, l2


def general_average_estimation(
        value_model: keras.Model,
        batch_s: np.ndarray,
        batch_r: rlearn.type.BatchReward,
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
        batch_r: rlearn.type.BatchReward,
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


def reshape_flat_gradients(
        grad_vars: tp.Dict[str, tp.Sequence[keras.Model]],
        gradients: np.ndarray,
) -> tp.Dict[str, tp.List[np.ndarray]]:
    assert gradients.ndim == 1, ValueError("grads must be 1d array")
    p = 0
    grads = {name: [] for name in grad_vars.keys()}
    keys = list(grads.keys())
    keys.sort()

    for g_key in keys:
        for model in grad_vars[g_key]:
            for w in model.weights:
                p_ = np.prod(w.shape) + p
                g = gradients[p: p_]
                grads[g_key].append(g.reshape(w.shape))
                p = p_
    return grads
