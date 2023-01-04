import os
import shutil
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLNet


class _PPO(BaseRLNet, metaclass=ABCMeta):
    is_on_policy = True

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

        self.pi = None
        self.pi_ = None
        self.critic = None

    @abstractmethod
    def dist(self, net, s):
        pass

    @staticmethod
    @abstractmethod
    def build_default_pi_callback(encoder: keras.Sequential, action_num: int):
        pass

    @staticmethod
    def build_default_critic_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def build(self, pi_encoder: keras.Model, critic_encoder: keras.Model, action_num: int):
        self.pi_ = self.build_default_pi_callback(pi_encoder, action_num)
        self.pi_._name = "pi_old"
        if self.training:
            self.pi = keras.models.clone_model(self.pi_)
            self.pi._name = "pi"
            self.critic = self.build_default_critic_callback(critic_encoder)
            self.critic._name = "critic"

    def predict(self, s):
        s = np.expand_dims(s, axis=0)
        dist = self.dist(self.pi_, s)  # use stable policy parameters to predict
        action = tf.squeeze(dist.sample(1)).numpy()
        if action.ndim == 0 and np.issubdtype(action, np.integer):
            action = int(action)
        return action

    def save(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.pi_.save_weights(os.path.join(model_tmp_dir, "pi_.ckpt"))
        self.critic.save_weights(os.path.join(model_tmp_dir, "critic.ckpt"))
        tools.zip_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.pi_.load_weights(os.path.join(unzipped_dir, "pi_.ckpt"))
        if self.training:
            self.pi.load_weights(os.path.join(unzipped_dir, "pi_.ckpt"))
            self.critic.load_weights(os.path.join(unzipped_dir, "critic.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    @staticmethod
    def build_critic_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class PPODiscrete(_PPO):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    def dist(self, net, s):
        o = net(s)
        return tfp.distributions.Categorical(probs=o)

    @staticmethod
    def build_default_pi_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        o = keras.layers.Softmax()(o)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class PPOContinue(_PPO):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    def dist(self, net, s):
        o = net(s)
        a_size = o.shape[1] // 2
        loc, scale = tf.tanh(o[:, :a_size]), tf.nn.sigmoid(o[:, a_size:])
        return tfp.distributions.Normal(loc=loc, scale=scale)

    @staticmethod
    def build_default_pi_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num * 2)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])
