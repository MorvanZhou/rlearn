"""
[Soft Actor-Critic: Off-Policy Maximum Entropy
 Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

The Soft Actor-Critic (SAC) algorithm extends the DDPG algorithm by
1) using a stochastic policy, which in theory can express multi-modal
optimal policies. This also enables the use of
2) entropy regularization based on the stochastic policy's entropy. It serves as a built-in,
state-dependent exploration heuristic for the agent, instead of relying on non-correlated
noise processes as in DDPG, or TD3 Additionally, it incorporates the
3) usage of two Soft Q-network to reduce the overestimation bias issue in Q-network-based methods.
"""

from abc import ABC

import tensorflow as tf
from tensorflow import keras

from rlearn.model.base import BaseStochasticModel


class _SAC(BaseStochasticModel, ABC):
    is_on_policy = False
    predicted_model_name = "actor"

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential, action_num: int):
        raise NotImplemented

    def set_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        a = self.set_actor_encoder_callback(actor, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic, action_num)
        self.set_model(a, c)

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.models["actor"] = actor
        if self.training:
            self.models["c1"] = critic
            self.models["c1_"] = self.clone_model(self.models["c1"])
            self.models["c2"] = critic
            self.models["c2_"] = self.clone_model(self.models["c2"])


class SACDiscrete(_SAC):
    name = __qualname__
    is_discrete_action = True

    def __init__(self, training: bool = True):
        super().__init__(training=training)

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class SACContinue(_SAC):
    name = __qualname__
    is_discrete_action = False

    def __init__(self, training: bool = True, ):
        super().__init__(training=training)

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential, action_num: int):
        action_inputs = keras.layers.InputLayer(
            input_shape=(action_num,),
            name="action_inputs",
            dtype=tf.float32)
        encoding = tf.concat([encoder.output, action_inputs.output], axis=1)
        o = keras.layers.Dense(action_num * 16)(encoding)
        o = keras.layers.ReLU()(o)
        o = keras.layers.Dense(32)(o)
        o = keras.layers.ReLU()(o)
        o = keras.layers.Dense(1)(o)
        return keras.Model(inputs=encoder.inputs + [action_inputs.input, ], outputs=[o])
