import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.sac import SACContinue, SACDiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer, TrainResult
from rlearn.trainer.tools import parse_2_learning_rate


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

    def set_default_optimizer(self):
        l1, l2 = parse_2_learning_rate(self.learning_rate)

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

    def train_batch(self) -> TrainResult:
        if self.opt_a is None or self.opt_c is None:
            self.set_default_optimizer()

        res = TrainResult(
            value={
                "actor_loss": 0,
                "critic_loss": 0,
                "reward": 0,
            },
            model_replaced=False,
        )
        if self.replay_buffer.is_empty():
            return res

        batch = self.replay_buffer.sample(self.batch_size)

        res.model_replaced = self.try_replace_params(
            [self.model.models["c1"], self.model.models["c2"]],
            [self.model.models["c1_"], self.model.models["c2_"]])

        with tf.GradientTape() as tape:
            if self.model.is_discrete_action:
                logits_ = self.model.models["actor"](batch["s_"])
                log_prob_ = tf.nn.log_softmax(logits_, axis=1)
                probs_ = tf.nn.softmax(logits_)

                q1_ = self.model.models["c1_"](batch["s_"])
                q2_ = self.model.models["c2_"](batch["s_"])
                q_min_ = tf.minimum(q1_, q2_)
                q_ = probs_ * (q_min_ - self.alpha * log_prob_)
                total_r = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
                non_terminate = 1. - batch["done"]
                q_ = total_r + self.gamma * tf.reduce_sum(q_, axis=1) * non_terminate

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
                    total_r = self.try_combine_int_ext_reward(batch["r"], batch["s_"])
                    non_terminate = (1. - batch["done"])[:, None]
                    q_ = total_r[:, None] + self.gamma * (q_a_min_ - self.alpha * log_prob_) * non_terminate
                q1_a = self.model.models["c1"]([batch["s"], batch["a"]])
                q2_a = self.model.models["c2"]([batch["s"], batch["a"]])

            lc = self.loss(q_, q1_a) + self.loss(q_, q2_a)

            tv = self.model.models["c1"].trainable_variables + self.model.models["c2"].trainable_variables
            grads = tape.gradient(lc, tv)
            self.opt_c.apply_gradients(zip(grads, tv))

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

            tv = self.model.models["actor"].trainable_variables
            grads = tape.gradient(la, tv)
            self.opt_a.apply_gradients(zip(grads, tv))

        res.value.update({
            "actor_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": total_r.mean(),
        })
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
