import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.config import TrainConfig
from rlearn.model.ppo import PPOContinue, PPODiscrete
from rlearn.model.tools import build_encoder_from_config
from rlearn.trainer import tools
from rlearn.trainer.base import BaseTrainer, TrainResult


class _PPOTrainer(BaseTrainer):

    def __init__(
            self,
            model: keras.Model,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            lam: float = 0.9,
            update_time: int = 1,
            use_gae: bool = True,
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
        self.use_gae = use_gae

        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_done = []

    def set_model_encoder_from_config(self, config: TrainConfig):
        action_num = len(config.action_transform)
        pi_encoder = build_encoder_from_config(config.nets[0], trainable=True)
        critic_encoder = build_encoder_from_config(config.nets[1], trainable=True)
        self.set_model_encoder(pi_encoder, critic_encoder, action_num)

    def set_default_optimizer(self):
        l1, l2 = tools.parse_2_learning_rate(self.learning_rate)

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
        self._set_tensorboard([self.model.models["pi"], self.model.models["critic"]])

    def set_model(self, pi: keras.Model, critic: keras.Model):
        self.model.set_model(pi=pi, critic=critic)
        self._set_tensorboard([self.model.models["pi"], self.model.models["critic"]])

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
            returns, adv = tools.general_average_estimation(
                value_model=self.model.models["critic"],
                batch_s=bs,
                batch_r=total_r,
                batch_done=self.buffer_done,
                s_=s_,
                gamma=self.gamma,
                lam=self.lam,
            )
            dist_ = self.model.dist(self.model.models["pi_"], bs)
            log_prob = dist_.log_prob(ba).numpy()
            self.replay_buffer.put_batch(
                s=bs,
                a=ba,
                returns=returns,
                adv=adv,
                old_log_prob=log_prob,
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
            adv = tools.discounted_adv(
                value_model=self.model.models["critic"],
                batch_s=bs,
                reward=returns,
            )
            dist_ = self.model.dist(self.model.models["pi_"], bs)
            log_prob = dist_.log_prob(ba).numpy()
            self.replay_buffer.put_batch(
                s=bs,
                a=ba,
                returns=returns,
                adv=adv,
                old_log_prob=log_prob,
            )

        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_done.clear()

    def train_batch(self) -> TrainResult:
        if self.opt_a is None or self.opt_c is None:
            self.set_default_optimizer()

        res = TrainResult(
            value={"pi_loss": 0, "critic_loss": 0, "reward": 0},
            model_replaced=False,
        )
        if self.replay_buffer.current_loading_point < self.batch_size:
            return res

        for _ in range(self.update_time):
            batch = self.replay_buffer.sample(self.batch_size)
            with tf.GradientTape() as tape:
                # critic
                vs = self.model.models["critic"](batch["s"])
                assert batch["returns"].ndim == 1, ValueError("batch['returns'].ndim != 1")
                lc = self.loss(batch["returns"][:, None], vs)

                tv = self.model.models["critic"].trainable_variables
                grads = tape.gradient(lc, tv)
                self.opt_c.apply_gradients(zip(grads, tv))

            with tf.GradientTape() as tape:
                # actor
                dist = self.model.dist(self.model.models["pi"], batch["s"])
                log_prob = dist.log_prob(batch["a"])

                assert batch["adv"].ndim == 1, ValueError("batch['adv'].ndim != 1")
                assert log_prob.ndim == 1, ValueError("log_prob.ndim != 1")
                assert batch["old_log_prob"].ndim == 1, ValueError("batch['old_log_prob'].ndim != 1")

                ratio = tf.exp(log_prob - batch["old_log_prob"])

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

                tv = self.model.models["pi"].trainable_variables
                grads = tape.gradient(la, tv)
                self.opt_a.apply_gradients(zip(grads, tv))

        res.model_replaced = self.try_replace_params(
            source=[self.model.models["pi"]], target=[self.model.models["pi_"]], ratio=1.)
        if res.model_replaced:
            self.replay_buffer.clear()
            self.buffer_s.clear()
            self.buffer_a.clear()
            self.buffer_r.clear()
            self.buffer_done.clear()

        res.value.update({
            "pi_loss": la.numpy(),
            "critic_loss": lc.numpy(),
            "reward": batch["returns"].mean(),
        })
        return res


class PPODiscreteTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
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


class PPOContinueTrainer(_PPOTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
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
