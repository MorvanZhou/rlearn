import os
import shutil
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLModel


class _ActorCritic(BaseRLModel, metaclass=ABCMeta):
    is_on_policy = True

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

        self.actor = None
        self.critic = None

    @abstractmethod
    def dist(self, net, s):
        pass

    @staticmethod
    @abstractmethod
    def set_actor_encoder_callback(encoder: keras.Sequential, action_num: int):
        pass

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def set_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        a = self.set_actor_encoder_callback(actor, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic)
        self.set_model(a, c)

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.actor = actor
        if self.training:
            self.critic = critic

    def predict(self, s):
        s = np.expand_dims(s, axis=0)
        dist = self.dist(self.actor, s)  # use stable policy parameters to predict
        action = tf.squeeze(dist.sample(1), axis=[0, 1]).numpy()
        if np.isnan(action).any():
            raise ValueError("action contains NaN")
        if action.ndim == 0 and np.issubdtype(action, np.integer):
            action = int(action)
        return action

    def disturbed_action(self, x, epsilon: float):
        raise NotImplemented

    def save_weights(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.actor.save_weights(os.path.join(model_tmp_dir, "actor.ckpt"))
        self.critic.save_weights(os.path.join(model_tmp_dir, "critic.ckpt"))
        tools.zip_ckpt_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.actor.load_weights(os.path.join(unzipped_dir, "actor.ckpt"))
        if self.training:
            self.critic.load_weights(os.path.join(unzipped_dir, "critic.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    def save(self, path: str):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.actor.save(os.path.join(model_tmp_dir, "actor"))
        self.critic.save(os.path.join(model_tmp_dir, "critic"))
        tools.zip_pb_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load(self, path: str):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.actor = keras.models.load_model(os.path.join(unzipped_dir, "actor"))
        if self.training:
            self.critic = keras.models.load_model(os.path.join(unzipped_dir, "critic"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    @staticmethod
    def build_critic_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class AcrtorCrititcDiscrete(_ActorCritic):
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
    def set_actor_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        o = keras.layers.Softmax()(o)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def disturbed_action(self, x, epsilon: float):
        return self.predict(x)


class AcrtorCrititcContinue(_ActorCritic):
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
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    @staticmethod
    def set_actor_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num * 2)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def disturbed_action(self, x, epsilon: float):
        return self.predict(x)
