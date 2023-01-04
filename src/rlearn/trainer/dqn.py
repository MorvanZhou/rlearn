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
            learning_rates: tp.Sequence[float],
            log_dir: str = None
    ):
        super().__init__(learning_rates, log_dir)

        self.model = DQN(training=True)
        self.opt = keras.optimizers.RMSprop(
            learning_rate=self.learning_rates[0],
            # global_clipnorm=2.,  # stable training
        )
        self.loss = keras.losses.MeanSquaredError()

    def build_model(self, net: keras.Model, action_num: int):
        self.model.build(net, action_num)
        self._set_tensorboard(self.model.net)

    def build_model_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        encoder = build_encoder_from_config(config.nets[0], trainable=True)
        self.build_model(encoder, action_num)

    def predict(self, s: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, sum(self.model.net.output_shape[1:]))
        return self.model.predict(s)

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.put_one(s, a, r, s_)

    def train_batch(self) -> tp.Dict[str, tp.Any]:
        res = {"loss": 0, "q": 0}
        if self.replay_buffer.empty():
            return res

        bs, ba, br, bs_ = self.replay_buffer.sample(self.batch_size)
        ba = ba.ravel().astype(np.int32)

        self.try_replace_params(src=self.model.net, target=self.model.net_)
        self.decay_epsilon()

        q_ = self.model.net_.predict(bs_, verbose=0)
        q_target = br.ravel() + self.gamma * tf.reduce_max(q_, axis=1)
        a_indices = tf.stack([tf.range(tf.shape(ba)[0], dtype=tf.int32), ba], axis=1)
        with tf.GradientTape() as tape:
            q = self.model.net(bs)
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                td = q_target - q_wrt_a
                loss = tf.reduce_mean(
                    tf.convert_to_tensor(self.replay_buffer.cache_importance_sampling_weights) * tf.square(td)
                )
                self.replay_buffer.batch_update(np.abs(td.numpy()))
            else:
                loss = self.loss(q_target, q_wrt_a)
        grads = tape.gradient(loss, self.model.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.net.trainable_variables))

        res.update({"loss": loss.numpy(), "q": q_wrt_a.numpy().ravel().mean()})
        return res

    def save_model(self, path: str):
        self.model.save(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)
