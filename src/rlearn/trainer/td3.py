import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.model.td3 import TD3
from rlearn.trainer import tools
from rlearn.trainer.base import TrainResult
from rlearn.trainer.ddpg import DDPGTrainer


class TD3Trainer(DDPGTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            exploration_std: float = 0.1,
            policy_std: float = 0.2,
            noise_clip: float = 0.1,
            policy_delay: int = 2,
    ):
        super().__init__(log_dir)
        self.model = TD3(training=True)
        self.opt_a, self.opt_c = None, None
        self.loss = keras.losses.MeanSquaredError()
        self.exploration_std = exploration_std
        self.policy_std = policy_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.train_count = 0

    def set_model_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        self.model.set_encoder(actor=actor, critic=critic, action_num=action_num)
        self._set_tensorboard([self.model.models["actor"], self.model.models["c1"]])

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.model.set_model(actor=actor, critic=critic)
        self._set_tensorboard([self.model.models["actor"], self.model.models["c1"]])

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
        grads["actor"]["v"] = self.model.models["actor"].trainable_variables
        grads["critic"]["v"] = self.model.models["c1"].trainable_variables + self.model.models["c2"].trainable_variables

        batch = self.replay_buffer.sample(self.batch_size)

        self.train_count += 1
        la = tf.convert_to_tensor(0)

        # update actor
        if self.train_count % self.policy_delay == 0:
            self.train_count = 0
            with tf.GradientTape() as tape:
                a = self.model.models["actor"](batch["s"])
                q = self.model.models["c1"]([batch["s"], a])
                la = tf.reduce_mean(-q)
                grads["actor"]["g"] = tape.gradient(la, grads["actor"]["v"])

        # update critic
        a_ = self.model.models["actor_"](batch["s_"])
        epsilon = tf.random.normal(a_.shape, mean=0.0, stddev=self.policy_std)
        clip_epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
        clip_a_ = tf.clip_by_value(a_ + clip_epsilon, -1, 1)

        q1_ = self.model.models["c1_"]([batch["s_"], clip_a_])
        q2_ = self.model.models["c2_"]([batch["s_"], clip_a_])
        min_q_ = tf.minimum(q1_, q2_)

        total_reward = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
        non_terminate = (1. - batch["done"])[:, None]

        q_ = total_reward[:, None] + self.gamma * min_q_ * non_terminate

        with tf.GradientTape() as tape:
            q1 = self.model.models["c1"]([batch["s"], batch["a"]])
            q2 = self.model.models["c2"]([batch["s"], batch["a"]])

            lc = self.replay_buffer.try_weighting_loss(target=q_, evaluated=q1)  # PR update only once
            lc += self.loss(q_, q2)

            grads["critic"]["g"] = tape.gradient(lc, grads["critic"]["v"])

        res.value.update({
            "actor_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": total_reward.mean(),
        })
        return res, grads

    def apply_flat_gradients(self, gradients: np.ndarray):
        assert gradients.dtype == np.float32, TypeError(f"gradients must be np.float32, but got {gradients.dtype}")
        a = self.model.models["actor"]
        c1 = self.model.models["c1"]
        c2 = self.model.models["c2"]
        reshaped_grads = tools.reshape_flat_gradients(
            grad_vars={"actor": [a], "critic": [c1, c2]},
            gradients=gradients,
        )
        if self.opt_a is None or self.opt_c is None:
            self._set_default_optimizer()
        self.opt_a.apply_gradients(zip(reshaped_grads["actor"], a.trainable_variables))
        self.opt_c.apply_gradients(zip(reshaped_grads["critic"], c1.trainable_variables + c2.trainable_variables))
        self.try_replace_params(
            [self.model.models["actor"], self.model.models["c1"], self.model.models["c2"]],
            [self.model.models["actor_"], self.model.models["c1_"], self.model.models["c2_"]]
        )

    def train_batch(self) -> TrainResult:
        res, grads = self.compute_gradients()
        if grads is not None:
            if self.opt_a is None or self.opt_c is None:
                self._set_default_optimizer()
            if len(grads["actor"]["g"]) != 0:
                self.opt_a.apply_gradients(zip(grads["actor"]["g"], grads["actor"]["v"]))
            self.opt_c.apply_gradients(zip(grads["critic"]["g"], grads["critic"]["v"]))
            res.model_replaced = self.try_replace_params(
                [self.model.models["actor"], self.model.models["c1"], self.model.models["c2"]],
                [self.model.models["actor_"], self.model.models["c1_"], self.model.models["c2_"]]
            )
        return res

    def predict(self, s: np.ndarray) -> np.ndarray:
        self.decay_epsilon()
        a = self.model.disturbed_action(s, self.epsilon)  # generate some random action
        epsilon = np.random.normal(0, self.exploration_std, size=a.shape)  # disturb action by norm
        a = np.clip(a + epsilon, -1., 1.)
        return a

    def store_transition(self, s, a, r, s_, done=False, *args, **kwargs):
        self.replay_buffer.put_one(s=s, a=a, r=r, s_=s_, done=done)
