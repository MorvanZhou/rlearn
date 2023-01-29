import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ac import ActorCriticContinue, ActorCriticDiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer, TrainResult
from rlearn.trainer.tools import parse_2_learning_rate


class _ActorCriticTrainer(BaseTrainer):
    is_on_policy = True

    def __init__(
            self,
            model: keras.Model,
            log_dir: str = None,
            entropy_coef: float = 0.01,
            lam: float = 0.9,
    ):
        super().__init__(log_dir)

        self.model: keras.Model = model
        self.opt_a = None
        self.opt_c = None

        self.lam = lam
        self.entropy_coef = entropy_coef

        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_done = []

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        actor_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(actor_encoder, critic_encoder, action_num)

    def set_default_optimizer(self):
        l1, l2 = parse_2_learning_rate(self.learning_rate)

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

        adv = []
        next_v = self.model.models["critic"].predict(
            np.expand_dims(np.array(s_, dtype=np.float32), axis=0),
            verbose=0).ravel()[0]

        bs = np.array(self.buffer_s, dtype=np.float32)
        vs = self.model.models["critic"].predict(bs, verbose=0).ravel()
        _gae_lam = 0
        for i in range(len(self.buffer_s) - 1, -1, -1):  # backward count
            non_terminate = 0. if self.buffer_done[i] else 1.
            delta = self.buffer_r[i] + self.gamma * next_v * non_terminate - vs[i]
            _gae_lam = delta + self.gamma * self.lam * _gae_lam * non_terminate
            adv.insert(0, _gae_lam)
            next_v = vs[i]
        ba = np.array(self.buffer_a, dtype=np.float32)
        adv = np.array(adv, dtype=np.float32)
        returns = adv + vs
        # adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        self.replay_buffer.put_batch(
            s=bs,
            a=ba,
            returns=np.expand_dims(returns, axis=1),
            # adv=adv,
        )

        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

    def train_batch(self) -> TrainResult:
        if self.opt_a is None or self.opt_c is None:
            self.set_default_optimizer()

        res = TrainResult(
            value={"actor_loss": 0, "critic_loss": 0},
            model_replaced=False,
        )
        if self.replay_buffer.current_loading_point < self.batch_size:
            return res

        batch = self.replay_buffer.sample(self.batch_size)
        with tf.GradientTape() as tape:
            # critic
            vs = self.model.models["critic"](batch["s"])
            td = batch["returns"] - vs
            lc = tf.reduce_mean(tf.square(td))

            tv = self.model.models["critic"].trainable_variables
            grads = tape.gradient(lc, tv)
            self.opt_c.apply_gradients(zip(grads, tv))

        with tf.GradientTape() as tape:
            # actor
            dist = self.model.dist(self.model.models["actor"], batch["s"])
            log_prob = dist.log_prob(batch["a"])
            exp_v = log_prob * tf.squeeze(td)

            if self.entropy_coef == 0.:
                entropy = 0.
            else:
                entropy = tf.reduce_mean(dist.entropy()) * self.entropy_coef

            la = - tf.reduce_mean(exp_v) - entropy

            tv = self.model.models["actor"].trainable_variables
            grads = tape.gradient(la, tv)
            self.opt_a.apply_gradients(zip(grads, tv))

        self.replay_buffer.clear()
        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

        res.value.update({"actor_loss": la.numpy(), "critic_loss": lc.numpy()})
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
