import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.dqn import DQN
from rlearn.model.tools import build_encoder_from_config
from rlearn.replaybuf.prioritized_buf import PrioritizedReplayBuffer
from rlearn.trainer.base import BaseTrainer, TrainResult


class DQNTrainer(BaseTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None
    ):
        super().__init__(log_dir)

        self.model = DQN(training=True)
        self.opt = None
        self.loss = keras.losses.MeanSquaredError()

    def set_default_optimizer(self):
        if isinstance(self.learning_rate, (tuple, list)):
            l1 = self.learning_rate[0]
        else:
            l1 = self.learning_rate

        self.opt = keras.optimizers.Adam(
            learning_rate=l1,
            # global_clipnorm=2.,  # stable training
        )

    def set_model_encoder(self, q: keras.Model, action_num: int):
        self.model.set_encoder(q, action_num)
        self._set_tensorboard(self.model.models["q"])

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        encoder = build_encoder_from_config(config.nets[0], trainable=True)
        self.set_model_encoder(encoder, action_num)

    def set_model(self, q: keras.Model):
        self.model.set_model(q=q)
        self._set_tensorboard([self.model.models["q"]])

    def predict(self, s: np.ndarray) -> int:
        self.decay_epsilon()
        return self.model.disturbed_action(s, self.epsilon)

    def store_transition(self, s, a, r, s_, done=False, *args, **kwargs):
        self.replay_buffer.put_one(s=s, a=a, r=r, s_=s_, done=done)

    def train_batch(self) -> TrainResult:
        if self.opt is None:
            self.set_default_optimizer()

        res = TrainResult(value={"loss": 0, "q": 0, "reward": 0}, model_replaced=False)
        if self.replay_buffer.is_empty():
            return res

        batch = self.replay_buffer.sample(self.batch_size)
        ba = batch["a"]

        res.model_replaced = self.try_replace_params(
            source=self.model.models["q"], target=self.model.models["q_"])

        q_ = self.model.models["q_"].predict(batch["s_"], verbose=0)
        total_reward = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
        assert total_reward.ndim == 1, ValueError("total_reward.ndim != 1")

        non_terminal = 1. - batch["done"]
        assert non_terminal.ndim == 1, ValueError("non_terminate.ndim != 1")

        q_target = total_reward + self.gamma * tf.reduce_max(q_, axis=1) * non_terminal

        a_indices = tf.stack([tf.range(tf.shape(ba)[0], dtype=tf.int32), ba], axis=1)
        with tf.GradientTape() as tape:
            q = self.model.models["q"](batch["s"])
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                td = q_target - q_wrt_a
                loss = tf.reduce_mean(
                    tf.convert_to_tensor(self.replay_buffer.cache_importance_sampling_weights) * tf.square(td)
                )
                self.replay_buffer.batch_update(np.abs(td.numpy()))
            else:
                loss = self.loss(q_target, q_wrt_a)
        tv = self.model.models["q"].trainable_variables
        grads = tape.gradient(loss, tv)
        self.opt.apply_gradients(zip(grads, tv))

        res.value.update({
            "loss": loss.numpy(),
            "q": q_wrt_a.numpy().ravel().mean(),
            "reward": total_reward.mean(),
        })
        return res
