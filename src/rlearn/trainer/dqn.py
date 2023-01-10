import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.dqn import DQN
from rlearn.model.tools import build_encoder_from_config
from rlearn.replaybuf.prioritized_buf import PrioritizedReplayBuffer
from rlearn.trainer.base import BaseTrainer


class DQNTrainer(BaseTrainer):
    name = __qualname__
    is_on_policy = False

    def __init__(
            self,
            log_dir: str = None
    ):
        super().__init__(log_dir)

        self.model = DQN(training=True)
        self.opt = None
        self.loss = keras.losses.MeanSquaredError()

    def set_default_optimizer(self):
        if isinstance(self.learning_rate, (tuple, list)) and len(self.learning_rate) <= 1:
            l_len = len(self.learning_rate)
            if l_len == 1:
                l1 = self.learning_rate[0]
            else:
                raise ValueError("learning rate must be 1")
        else:
            l1 = self.learning_rate

        self.opt = keras.optimizers.Adam(
            learning_rate=l1,
            # global_clipnorm=2.,  # stable training
        )

    def set_model_encoder(self, q: keras.Model, action_num: int):
        self.model.set_encoder(q, action_num)
        self._set_tensorboard(self.model.q)

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        encoder = build_encoder_from_config(config.nets[0], trainable=True)
        self.set_model_encoder(encoder, action_num)

    def set_model(self, q: keras.Model):
        self.model.set_model(q=q)
        self._set_tensorboard([self.model.q])

    def predict(self, s: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, sum(self.model.q.output_shape[1:]))
        return self.model.predict(s)

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.put_one(s, a, r, s_)

    def train_batch(self) -> tp.Dict[str, tp.Any]:
        if self.opt is None:
            self.set_default_optimizer()

        res = {"loss": 0, "q": 0}
        if self.replay_buffer.empty():
            return res

        bs, ba, br, bs_ = self.replay_buffer.sample(self.batch_size)
        ba = ba.ravel().astype(np.int32)

        replaced = self.try_replace_params(src=self.model.q, target=self.model.q_)
        self.decay_epsilon()

        q_ = self.model.q_.predict(bs_, verbose=0)
        q_target = br.ravel() + self.gamma * tf.reduce_max(q_, axis=1)
        a_indices = tf.stack([tf.range(tf.shape(ba)[0], dtype=tf.int32), ba], axis=1)
        with tf.GradientTape() as tape:
            q = self.model.q(bs)
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                td = q_target - q_wrt_a
                loss = tf.reduce_mean(
                    tf.convert_to_tensor(self.replay_buffer.cache_importance_sampling_weights) * tf.square(td)
                )
                self.replay_buffer.batch_update(np.abs(td.numpy()))
            else:
                loss = self.loss(q_target, q_wrt_a)
        grads = tape.gradient(loss, self.model.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.q.trainable_variables))

        res.update({"loss": loss.numpy(), "q": q_wrt_a.numpy().ravel().mean(), "replaced": replaced})
        return res

    def save_model_weights(self, path: str):
        self.model.save_weights(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model.load(path)
