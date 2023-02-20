import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ddpg import DDPG
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer import tools
from rlearn.trainer.base import BaseTrainer, TrainResult


class DDPGTrainer(BaseTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None
    ):
        super().__init__(log_dir)
        self.model = DDPG(training=True)
        self.opt_a, self.opt_c = None, None

    def _set_default_optimizer(self):
        l1, l2 = tools.parse_2_learning_rate(self.learning_rate)

        self.opt_a = keras.optimizers.Adam(
            learning_rate=l1,
        )
        self.opt_c = keras.optimizers.Adam(
            learning_rate=l2,
        )

    def set_model_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        self.model.set_encoder(actor=actor, critic=critic, action_num=action_num)
        self._set_tensorboard([self.model.models["actor"], self.model.models["critic"]])

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.model.set_model(actor=actor, critic=critic)
        self._set_tensorboard([self.model.models["actor"], self.model.models["critic"]])

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        actor_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(actor=actor_encoder, critic=critic_encoder, action_num=action_num)

    def compute_gradients(self) -> tp.Tuple[TrainResult, tp.Optional[tp.Dict[str, tp.Dict[str, list]]]]:
        res = TrainResult(
            value={
                "actor_loss": 0,
                "critic_loss": 0,
                "reward": 0,
            },
            model_replaced=False,
        )

        if self.replay_buffer.is_empty():
            return res, None

        grads = {"actor": {"g": [], "v": []}, "critic": {"g": [], "v": []}}
        batch = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            a = self.model.models["actor"](batch["s"])
            q = self.model.models["critic"]([batch["s"], a])
            la = tf.reduce_mean(-q)

            tv = self.model.models["actor"].trainable_variables
            grads["actor"]["g"] = tape.gradient(la, tv)
            grads["actor"]["v"] = tv

        with tf.GradientTape() as tape:
            with tape.stop_recording():
                a_ = self.model.models["actor_"](batch["s_"])
                total_reward = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
                non_terminal = (1 - batch["done"])[:, None]
                assert non_terminal.ndim == 2, ValueError("non_terminal.ndim != 2")
                assert total_reward.ndim == 1, ValueError("total_reward.ndim != 1")

                va_ = self.model.models["critic_"]([batch["s_"], a_])
                q_ = total_reward[:, None] + self.gamma * va_ * non_terminal
            q = self.model.models["critic"]([batch["s"], batch["a"]])
            lc = self.replay_buffer.try_weighting_loss(target=q_, evaluated=q)

            tv = self.model.models["critic"].trainable_variables
            grads["critic"]["g"] = tape.gradient(lc, tv)
            grads["critic"]["v"] = tv

        res.value.update({
            "actor_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": total_reward.mean(),
        })
        return res, grads

    def apply_flat_gradients(self, gradients: np.ndarray):
        assert gradients.dtype == np.float32, TypeError(f"gradients must be np.float32, but got {gradients.dtype}")
        a = self.model.models["actor"]
        c = self.model.models["critic"]
        reshaped_grads = tools.reshape_flat_gradients(
            grad_vars={"actor": [a], "critic": [c]},
            gradients=gradients,
        )

        if self.opt_a is None or self.opt_c is None:
            self._set_default_optimizer()
        self.opt_a.apply_gradients(zip(reshaped_grads["actor"], a.trainable_variables))
        self.opt_c.apply_gradients(zip(reshaped_grads["critic"], c.trainable_variables))
        self.try_replace_params(
            [self.model.models["actor"], self.model.models["critic"]],
            [self.model.models["actor_"], self.model.models["critic_"]]
        )

    def train_batch(self) -> TrainResult:
        res, grads = self.compute_gradients()
        if grads is not None:
            if self.opt_a is None or self.opt_c is None:
                self._set_default_optimizer()
            self.opt_a.apply_gradients(zip(grads["actor"]["g"], grads["actor"]["v"]))
            self.opt_c.apply_gradients(zip(grads["critic"]["g"], grads["critic"]["v"]))
            res.model_replaced = self.try_replace_params(
                [self.model.models["actor"], self.model.models["critic"]],
                [self.model.models["actor_"], self.model.models["critic_"]]
            )
        return res

    def predict(self, s: np.ndarray) -> np.ndarray:
        self.decay_epsilon()
        return self.model.disturbed_action(s, self.epsilon)

    def store_transition(self, s, a, r, s_, done=False, *args, **kwargs):
        self.replay_buffer.put_one(s=s, a=a, r=r, s_=s_, done=done)
