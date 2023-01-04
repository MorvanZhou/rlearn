import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ppo import PPOContinue, PPODiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer


class _PPOTrainer(BaseTrainer):
    is_on_policy = True

    def __init__(
            self,
            model: keras.Model,
            learning_rates: tp.Sequence[float],
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            update_time: int = 1,
    ):
        super().__init__(learning_rates, log_dir)
        if len(learning_rates) != 2:
            raise ValueError("length of learning rate for ddpg must be 2, (actor's and critic's)")

        self.model: keras.Model = model
        self.opt_a = keras.optimizers.Adam(
            learning_rate=self.learning_rates[0],
            global_clipnorm=2.,  # stable training
        )
        self.opt_c = keras.optimizers.Adam(
            learning_rate=self.learning_rates[1],
            global_clipnorm=2.,  # stable training
        )
        self.loss = keras.losses.MeanSquaredError()

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.update_time = update_time

        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []

    def build_model(self, pi_encoder: keras.Model, critic_encoder: keras.Model, action_num: int):
        self.model.build(pi_encoder, critic_encoder, action_num)
        self._set_tensorboard([self.model.pi, self.model.critic])

    def build_model_from_config(self, config: TrainConfig):
        raise NotImplemented

    def predict(self, s: np.ndarray) -> np.ndarray:
        return self.model.predict(s)

    # def update_policy(self):
    #     self.replace_target_net(src=self.model.pi, target=self.model.pi_, ratio=1.)
    #     self.replay_buffer.clear()

    def compute_vs(self, s):
        s = np.expand_dims(s, axis=0)
        vs = self.model.critic.predict(s, verbose=0)
        vs = vs.ravel()[0]
        return vs

    def store_transition(self, s, a, r, done: bool):
        self.buffer_s.append(s)
        self.buffer_a.append([a] if isinstance(a, (int, float, np.int32, np.int64, np.int8)) else a)
        self.buffer_r.append(r)
        if not done and len(self.buffer_s) < self.batch_size:
            return

        discounted_r = []

        s_ = self.buffer_s[-1]
        vs_ = self.compute_vs(s_)
        for i in range(len(self.buffer_s) - 1, -1, -1):  # backward count
            vs_ = self.buffer_r[i] + self.gamma * vs_
            discounted_r.append(vs_)
        ba = np.array(self.buffer_a, dtype=np.float32)
        if ba.ndim == 1:
            ba = ba[:, None]
        self.replay_buffer.put_batch(
            s=np.vstack(self.buffer_s),
            a=ba,
            r=np.array(discounted_r[::-1], dtype=np.float32),
            s_=None,
        )
        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()

    def train_batch(self) -> tp.Dict[str, tp.Any]:
        res = {"a_loss": 0, "c_loss": 0}
        if self.replay_buffer.empty():
            return res

        for _ in range(self.update_time):
            bs, ba, br, _ = self.replay_buffer.sample(self.batch_size)

            with tf.GradientTape() as tape:
                # critic
                vs = self.model.critic(bs)
                lc = self.loss(br, vs)

                grads = tape.gradient(lc, self.model.critic.trainable_variables)
                self.opt_a.apply_gradients(zip(grads, self.model.critic.trainable_variables))

            with tf.GradientTape() as tape:
                # actor
                advantage = br - vs
                dist_ = self.model.dist(self.model.pi_, bs)
                dist = self.model.dist(self.model.pi, bs)
                ratio = tf.exp(dist.log_prob(ba) - dist_.log_prob(ba))
                surrogate = ratio * advantage
                clipped_surrogate = tf.clip_by_value(
                    ratio, 1. - self.clip_epsilon, 1. + self.clip_epsilon
                ) * advantage
                entropy = 0. if self.entropy_coef == 0. else tf.reduce_mean(dist.entropy()) * self.entropy_coef

                la = - tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate)) - entropy

                grads = tape.gradient(la, self.model.pi.trainable_variables)
                self.opt_a.apply_gradients(zip(grads, self.model.pi.trainable_variables))

        replaced = self.try_replace_params(src=self.model.pi, target=self.model.pi_, ratio=1.)
        if replaced:
            self.replay_buffer.clear()

        res.update({"a_loss": la.numpy(), "c_loss": lc.numpy()})
        return res

    def save_model(self, path: str):
        self.model.save(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)


class PPODiscreteTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            learning_rates: tp.Sequence[float],
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            update_time: int = 1,
    ):
        super().__init__(PPODiscrete(training=True), learning_rates, log_dir, clip_epsilon, entropy_coef, update_time)

    def build_model_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)

        pi_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.build_model(pi_encoder, critic_encoder, action_num)


class PPOContinueTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            learning_rates: tp.Sequence[float],
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            update_time: int = 1,
    ):
        super().__init__(PPOContinue(training=True), learning_rates, log_dir, clip_epsilon, entropy_coef, update_time)

    def build_model_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        pi_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.build_model(pi_encoder, critic_encoder, action_num)
