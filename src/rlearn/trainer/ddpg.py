import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ddpg import DDPG
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer


class DDPGTrainer(BaseTrainer):
    name = __qualname__
    is_on_policy = False

    def __init__(
            self,
            learning_rates: tp.Sequence[float],
            log_dir: str = None
    ):
        super().__init__(learning_rates, log_dir)

        if len(learning_rates) != 2:
            raise ValueError("length of learning rate for ddpg must be 2, (actor's and critic's)")
        self.model = DDPG(training=True)
        self.opt_a = keras.optimizers.Adam(
            learning_rate=self.learning_rates[0],
            # global_clipnorm=2.,  # stable training
        )
        self.opt_c = keras.optimizers.Adam(
            learning_rate=self.learning_rates[1],
            # global_clipnorm=2.,  # stable training
        )
        self.loss = keras.losses.MeanSquaredError()

    def build_model(self, actor_encoder: keras.Model, critic_encoder: keras.Model, action_num: int):
        self.model.build(actor_encoder=actor_encoder, critic_encoder=critic_encoder, action_num=action_num)
        self._set_tensorboard([self.model.actor, self.model.critic])

    def build_model_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        actor_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.build_model(actor_encoder=actor_encoder, critic_encoder=critic_encoder, action_num=action_num)

    def train_batch(self) -> tp.Dict[str, tp.Any]:
        res = {
            "a_loss": 0,
            "c_loss": 0,
        }
        if self.replay_buffer.empty():
            return res

        bs, ba, br, bs_ = self.replay_buffer.sample(self.batch_size)

        self.try_replace_params(self.model.actor, self.model.actor_)
        self.try_replace_params(self.model.critic, self.model.critic_)
        self.decay_epsilon()

        with tf.GradientTape() as tape:
            a = self.model.actor(bs)
            q = self.model.critic([bs, a])
            la = tf.reduce_mean(-q)

            grads = tape.gradient(la, self.model.actor.trainable_variables)
            self.opt_a.apply_gradients(zip(grads, self.model.actor.trainable_variables))

        with tf.GradientTape() as tape:
            a_ = self.model.actor_(bs_)
            q_ = br + self.gamma * self.model.critic_([bs_, a_])
            q = self.model.critic([bs, ba])
            lc = self.loss(q_, q)

            grads = tape.gradient(lc, self.model.critic.trainable_variables)
            self.opt_c.apply_gradients(zip(grads, self.model.critic.trainable_variables))

        res.update({
            "a_loss": la.numpy(),
            "c_loss": lc.numpy(),
        })
        return res

    def predict(self, s: np.ndarray) -> np.ndarray:
        if np.random.random() < self.epsilon:
            return np.random.uniform(-1, 1, size=sum(self.model.actor.output_shape[1:])).astype(np.float32)
        return self.model.predict(s)

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.put_one(s, a, r, s_)

    def save_model(self, path: str):
        self.model.save(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)
