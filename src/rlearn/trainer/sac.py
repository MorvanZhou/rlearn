import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.sac import SACContinue, SACDiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer import tools
from rlearn.trainer.base import BaseTrainer, TrainResult


class _SACTrainer(BaseTrainer):
    name = __qualname__

    def __init__(
            self,
            model: keras.Model,
            log_dir: str = None,
            alpha: float = 0.2,
    ):
        super().__init__(log_dir)
        self.model: keras.Model = model
        self.alpha = alpha
        self.opt_a, self.opt_c = None, None
        self.loss = keras.losses.MeanSquaredError()

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
        self._set_tensorboard([self.model.models["actor"], self.model.models["c1"]])

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.model.set_model(actor=actor, critic=critic)
        self._set_tensorboard([self.model.models["actor"], self.model.models["c1"]])

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

        grads = {"critic": {"g": [], "v": []}, "actor": {"g": [], "v": []}}
        grads["actor"]["v"] = self.model.models["actor"].trainable_variables
        grads["critic"]["v"] = self.model.models["c1"].trainable_variables + self.model.models["c2"].trainable_variables

        batch = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape() as tape:
            total_r = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
            non_terminal = (1. - batch["done"])
            assert non_terminal.ndim == 1, ValueError("non_terminal.ndim != 1")
            assert total_r.ndim == 1, ValueError("total_reward.ndim != 1")

            if self.model.is_discrete_action:
                with tape.stop_recording():
                    logits_ = self.model.models["actor"](batch["s_"])
                    log_prob_ = tf.nn.log_softmax(logits_, axis=1)
                    probs_ = tf.nn.softmax(logits_)

                    q1_ = self.model.models["c1_"](batch["s_"])
                    q2_ = self.model.models["c2_"](batch["s_"])
                    q_min_ = tf.minimum(q1_, q2_)
                    q_ = probs_ * (q_min_ - self.alpha * log_prob_)
                    q_ = total_r + self.gamma * tf.reduce_sum(q_, axis=1) * non_terminal
                    a_indices = tf.stack(
                        [tf.range(tf.shape(batch["a"])[0], dtype=tf.int32), batch["a"]], axis=1)

                q1 = self.model.models["c1"](batch["s"])
                q2 = self.model.models["c2"](batch["s"])
                q1_a = tf.gather_nd(params=q1, indices=a_indices)
                q2_a = tf.gather_nd(params=q2, indices=a_indices)
            else:
                with tape.stop_recording():
                    dist_ = self.model.dist(self.model.models["actor"], batch["s_"])
                    a_ = tf.squeeze(dist_.sample(1), axis=0)
                    log_prob_ = tf.expand_dims(dist_.log_prob(a_), axis=1)
                    q1_a_ = self.model.models["c1_"]([batch["s_"], a_])
                    q2_a_ = self.model.models["c2_"]([batch["s_"], a_])
                    q_a_min_ = tf.minimum(q1_a_, q2_a_)
                    q_ = total_r[:, None] + self.gamma * (q_a_min_ - self.alpha * log_prob_) * non_terminal[:, None]
                q1_a = self.model.models["c1"]([batch["s"], batch["a"]])
                q2_a = self.model.models["c2"]([batch["s"], batch["a"]])

            lc = self.replay_buffer.try_weighting_loss(target=q_, evaluated=q1_a)  # PR update only once
            lc += self.loss(q_, q2_a)
            grads["critic"]["g"] = tape.gradient(lc, grads["critic"]["v"])

        with tf.GradientTape() as tape:
            # kl divergence
            if self.model.is_discrete_action:
                logits = self.model.models["actor"](batch["s"])
                log_prob = tf.nn.log_softmax(logits, axis=1)
                probs = tf.nn.softmax(logits)
                # q1 = self.model.models["c1"](batch["s"])
                # q2 = self.model.models["c2"](batch["s"])
                q_min = tf.minimum(q1, q2)
                la = tf.reduce_mean(probs * (self.alpha * log_prob - q_min))
            else:
                dist = self.model.dist(self.model.models["actor"], batch["s"])
                a = tf.squeeze(dist.sample(1), axis=0)
                log_prob = tf.expand_dims(dist.log_prob(a), axis=1)
                q1_a = self.model.models["c1"]([batch["s"], a])
                q2_a = self.model.models["c2"]([batch["s"], a])
                q_a_min = tf.minimum(q1_a, q2_a)
                la = tf.reduce_mean(self.alpha * log_prob - q_a_min)

            grads["actor"]["g"] = tape.gradient(la, grads["actor"]["v"])

        res.value.update({
            "actor_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": total_r.mean(),
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
            [self.model.models["c1"], self.model.models["c2"]],
            [self.model.models["c1_"], self.model.models["c2_"]]
        )

    def train_batch(self) -> TrainResult:
        res, grads = self.compute_gradients()
        if grads is not None:
            if self.opt_a is None or self.opt_c is None:
                self._set_default_optimizer()
            self.opt_c.apply_gradients(zip(grads["critic"]["g"], grads["critic"]["v"]))
            self.opt_a.apply_gradients(zip(grads["actor"]["g"], grads["actor"]["v"]))
            res.model_replaced = self.try_replace_params(
                [self.model.models["c1"], self.model.models["c2"]],
                [self.model.models["c1_"], self.model.models["c2_"]]
            )
        return res

    def predict(self, s: np.ndarray) -> np.ndarray:
        self.decay_epsilon()
        return self.model.disturbed_action(s, self.epsilon)

    def store_transition(self, s, a, r, s_, done=False, *args, **kwargs):
        self.replay_buffer.put_one(s=s, a=a, r=r, s_=s_, done=done)


class SACDiscreteTrainer(_SACTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            alpha: float = 0.2,
    ):
        super().__init__(
            SACDiscrete(training=True),
            log_dir=log_dir,
            alpha=alpha,
        )


class SACContinueTrainer(_SACTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            alpha: float = 0.2,
    ):
        super().__init__(
            SACContinue(training=True),
            log_dir=log_dir,
            alpha=alpha,
        )
