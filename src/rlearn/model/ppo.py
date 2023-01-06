import os
import shutil
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLModel


class _PPO(BaseRLModel, metaclass=ABCMeta):
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
    def set_pi_encoder_callback(encoder: keras.Sequential, action_num: int):
        pass

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def set_encoder(self, pi: keras.Model, critic: keras.Model, action_num: int):
        old_pi = self.set_pi_encoder_callback(pi, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic)
        self.set_model(old_pi, c)

    def set_model(self, pi: keras.Model, critic: keras.Model):
        self.pi_ = pi
        if self.training:
            self.pi = self.clone_model(self.pi_)
            self.critic = critic

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
    def set_pi_encoder_callback(encoder: keras.Sequential, action_num: int):
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
    def set_pi_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num * 2)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])
