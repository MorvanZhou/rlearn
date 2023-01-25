import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ppo import PPOContinue, PPODiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer.base import BaseTrainer, TrainResult
from rlearn.trainer.tools import parse_2_learning_rate


class _PPOTrainer(BaseTrainer):
    is_on_policy = True

    def __init__(
            self,
            model: keras.Model,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            lam: float = 0.9,
            update_time: int = 1,
    ):
        super().__init__(log_dir)

        self.model: keras.Model = model
        self.opt_a = None
        self.opt_c = None
        self.loss = keras.losses.MeanSquaredError()

        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.update_time = update_time

        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_done = []

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

    def set_model_encoder(self, pi: keras.Model, critic: keras.Model, action_num: int):
        self.model.set_encoder(pi, critic, action_num)
        self._set_tensorboard([self.model.pi, self.model.critic])

    def set_model_encoder_from_config(self, config: TrainConfig):
        raise NotImplemented

    def set_model(self, pi: keras.Model, critic: keras.Model):
        self.model.set_model(pi=pi, critic=critic)
        self._set_tensorboard([self.model.pi, self.model.critic])

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
        next_v = self.model.critic.predict(
            np.expand_dims(np.array(s_, dtype=np.float32), axis=0),
            verbose=0).ravel()[0]

        bs = np.array(self.buffer_s, dtype=np.float32)
        vs = self.model.critic.predict(bs, verbose=0).ravel()
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        dist_ = self.model.dist(self.model.pi_, bs)
        log_prob = dist_.log_prob(ba).numpy()
        self.replay_buffer.put_batch(
            s=bs,
            a=ba,
            returns=np.expand_dims(returns, axis=1),
            adv=adv,
            old_log_prob=log_prob,
        )

        # discounted_r = []
        # vs_ = self.model.critic.predict(
        #     np.expand_dims(np.array(s_, dtype=np.float32), axis=0),
        #     verbose=0).ravel()[0]
        # for i in range(len(self.buffer_s) - 1, -1, -1):  # backward count
        #     if self.buffer_done[i]:
        #         vs_ = 0
        #     vs_ = self.buffer_r[i] + self.gamma * vs_
        #     discounted_r.insert(0, vs_)
        # bs = np.vstack(self.buffer_s)
        # ba = np.array(self.buffer_a, dtype=np.float32)
        # dist_ = self.model.dist(self.model.pi_, bs)
        # log_prob = dist_.log_prob(ba).numpy()
        # returns = np.array(discounted_r, dtype=np.float32)[:, None]
        # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # adv = (returns - self.model.critic.predict(bs, verbose=0)).ravel()
        # self.replay_buffer.put_batch(
        #     s=bs,
        #     a=ba,
        #     returns=returns,
        #     adv=adv,
        #     old_log_prob=log_prob,
        # )

        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

    def train_batch(self) -> TrainResult:
        if self.opt_a is None or self.opt_c is None:
            self.set_default_optimizer()

        res = TrainResult(
            value={"pi_loss": 0, "critic_loss": 0},
            model_replaced=False,
        )
        if self.replay_buffer.current_loading_point < self.batch_size:
            return res

        for _ in range(self.update_time):
            batch = self.replay_buffer.sample(self.batch_size)
            with tf.GradientTape() as tape:
                # critic
                vs = self.model.critic(batch["s"])
                lc = self.loss(batch["returns"], vs)

                grads = tape.gradient(lc, self.model.critic.trainable_variables)
                self.opt_c.apply_gradients(zip(grads, self.model.critic.trainable_variables))

            with tf.GradientTape() as tape:
                # actor
                dist = self.model.dist(self.model.pi, batch["s"])
                ratio = tf.exp(dist.log_prob(batch["a"]) - batch["old_log_prob"])
                surrogate = ratio * batch["adv"]
                clipped_surrogate = tf.clip_by_value(
                    ratio,
                    1. - self.clip_epsilon,
                    1. + self.clip_epsilon
                ) * batch["adv"]
                if self.entropy_coef == 0.:
                    entropy = 0.
                else:
                    entropy = tf.reduce_mean(dist.entropy()) * self.entropy_coef

                la = - tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate)) - entropy

                grads = tape.gradient(la, self.model.pi.trainable_variables)
                self.opt_a.apply_gradients(zip(grads, self.model.pi.trainable_variables))

        res.model_replaced = self.try_replace_params(
            source=[self.model.pi], target=[self.model.pi_], ratio=1.)
        if res.model_replaced:
            self.replay_buffer.clear()
            self.buffer_s.clear()
            self.buffer_a.clear()
            self.buffer_r.clear()
            self.buffer_done.clear()

        res.value.update({"pi_loss": la.numpy(), "critic_loss": lc.numpy()})
        return res

    def save_model_weights(self, path: str):
        self.model.save_weights(path)

    def load_model_weights(self, path: str):
        self.model.load_weights(path)

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        self.model.load(path)


class PPODiscreteTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            lam: float = 0.9,
            update_time: int = 1,
    ):
        super().__init__(
            PPODiscrete(training=True),
            log_dir=log_dir,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            lam=lam,
            update_time=update_time,
        )

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)

        pi_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(pi_encoder, critic_encoder, action_num)


class PPOContinueTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.,
            lam: float = 0.9,
            update_time: int = 1,
    ):
        super().__init__(
            PPOContinue(training=True),
            log_dir=log_dir,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            lam=lam,
            update_time=update_time,
        )

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        pi_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(pi_encoder, critic_encoder, action_num)
