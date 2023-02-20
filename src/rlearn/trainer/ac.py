import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ac import ActorCriticContinue, ActorCriticDiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer import tools
from rlearn.trainer.base import BaseTrainer, TrainResult


class _ActorCriticTrainer(BaseTrainer):

    def __init__(
            self,
            model: keras.Model,
            log_dir: str = None,
            entropy_coef: float = 0.01,
            lam: float = 0.9,
            use_gae: bool = True,
    ):
        super().__init__(log_dir)

        self.model: keras.Model = model
        self.opt_a = None
        self.opt_c = None

        self.lam = lam
        self.entropy_coef = entropy_coef
        self.use_gae = use_gae

        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_done = []

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        actor_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(actor_encoder, critic_encoder, action_num)

    def _set_default_optimizer(self):
        l1, l2 = tools.parse_2_learning_rate(self.learning_rate)

        self.opt_a = keras.optimizers.Adam(
            learning_rate=l1,
            # global_clipnorm=5.,  # stable training
        )
        self.opt_c = keras.optimizers.Adam(
            learning_rate=l2,
            # global_clipnorm=5.,  # stable training
        )

    def set_model_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        self.model.set_encoder(actor, critic, action_num)
        self._set_tensorboard([self.model.models["actor"], self.model.models["critic"]])

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.model.set_model(actor=actor, critic=critic)
        self._set_tensorboard([self.model.models["actor"], self.model.models["critic"]])

    def predict(self, s: np.ndarray) -> np.ndarray:
        self.decay_epsilon()
        return self.model.disturbed_action(s, self.epsilon)

    def store_transition(self, s, a, r, s_, done: bool = False):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_done.append(done)
        if len(self.buffer_s) < self.batch_size:
            return

        next_s = np.array(self.buffer_s[1:] + [s_])
        total_r = self.try_combine_int_ext_reward(self.buffer_r, next_s)
        ba = np.array(self.buffer_a, dtype=np.float32)
        bs = np.array(self.buffer_s, dtype=np.float32)

        if self.use_gae:
            returns, _ = tools.general_average_estimation(
                value_model=self.model.models["critic"],
                batch_s=bs,
                batch_r=total_r,
                batch_done=self.buffer_done,
                s_=s_,
                gamma=self.gamma,
                lam=self.lam,
            )

            self.replay_buffer.put_batch(
                s=bs,
                a=ba,
                returns=returns,
                # adv=adv,
            )
        else:
            returns = tools.discounted_reward(
                value_model=self.model.models["critic"],
                batch_s=bs,
                batch_r=total_r,
                batch_done=self.buffer_done,
                s_=s_,
                gamma=self.gamma,
            )
            self.replay_buffer.put_batch(
                s=bs,
                a=ba,
                returns=returns,
            )

        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

    def compute_gradients(self) -> tp.Tuple[TrainResult, tp.Optional[tp.Dict[str, tp.Dict[str, list]]]]:
        res = TrainResult(
            value={"actor_loss": 0., "critic_loss": 0., "reward": 0.},
            model_replaced=False,
        )
        grads = {"actor": {"g": [], "v": []}, "critic": {"g": [], "v": []}}
        if self.replay_buffer.current_loading_point < self.batch_size:
            return res, None

        batch = self.replay_buffer.sample(self.batch_size)
        with tf.GradientTape() as tape:
            # critic
            vs = self.model.models["critic"](batch["s"])
            assert batch["returns"].ndim == 1, ValueError("batch['returns'].ndim != 1")
            adv = batch["returns"][:, None] - vs
            lc = tf.reduce_mean(tf.square(adv))

            tv = self.model.models["critic"].trainable_variables
            grads["critic"]["g"] = tape.gradient(lc, tv)
            grads["critic"]["v"] = tv

        with tf.GradientTape() as tape:
            # actor
            dist = self.model.dist(self.model.models["actor"], batch["s"])
            log_prob = dist.log_prob(batch["a"])
            adv = tf.squeeze(adv)
            assert adv.ndim == 1, ValueError("adv.ndim != 1")
            assert log_prob.ndim == 1, ValueError("log_prob.ndim != 1")
            exp_v = log_prob * tf.squeeze(adv)

            if self.entropy_coef == 0.:
                entropy = 0.
            else:
                entropy = tf.reduce_mean(dist.entropy()) * self.entropy_coef

            la = - tf.reduce_mean(exp_v) - entropy

            tv = self.model.models["actor"].trainable_variables
            grads["actor"]["g"] = tape.gradient(la, tv)
            grads["actor"]["v"] = tv

        self.replay_buffer.clear()
        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

        res.value.update({
            "actor_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": batch["returns"].mean(),
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

    def train_batch(self) -> TrainResult:
        res, grads = self.compute_gradients()
        if grads is not None:
            if self.opt_a is None or self.opt_c is None:
                self._set_default_optimizer()
            self.opt_a.apply_gradients(zip(grads["actor"]["g"], grads["actor"]["v"]))
            self.opt_c.apply_gradients(zip(grads["critic"]["g"], grads["critic"]["v"]))
        return res


class ActorCriticDiscreteTrainer(_ActorCriticTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            entropy_coef: float = 0.01,
            lam: float = 0.9,
    ):
        super().__init__(
            ActorCriticDiscrete(training=True),
            log_dir=log_dir,
            entropy_coef=entropy_coef,
            lam=lam,
        )


class ActorCriticContinueTrainer(_ActorCriticTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            entropy_coef: float = 0.01,
            lam: float = 0.9,
    ):
        super().__init__(
            ActorCriticContinue(training=True),
            log_dir=log_dir,
            entropy_coef=entropy_coef,
            lam=lam,
        )
