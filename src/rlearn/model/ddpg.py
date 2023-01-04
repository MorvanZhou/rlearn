import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLNet


class DDPG(BaseRLNet):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        BaseRLNet.__init__(self, training=training)
        self.critic = None
        self.critic_ = None
        self.actor = None
        self.actor_ = None

    @staticmethod
    def default_build_actor_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num, activation="tanh")(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    @staticmethod
    def default_build_critic_callback(encoder: keras.Sequential, action_num: int):
        action_inputs = keras.layers.InputLayer(
            input_shape=(action_num,),
            name="action_inputs",
            dtype=tf.float32)
        encoding = tf.concat([encoder.output, action_inputs.output], axis=1)
        o = keras.layers.Dense(action_num * 16)(encoding)
        o = keras.layers.ReLU()(o)
        o = keras.layers.Dense(1)(o)
        return keras.Model(inputs=encoder.inputs + [action_inputs.input, ], outputs=[o])

    def build(self, actor_encoder: keras.Model, critic_encoder: keras.Model, action_num: int):
        self.actor = self.default_build_actor_callback(actor_encoder, action_num)
        self.actor._name = "actor"
        if self.training:
            self.actor_ = keras.models.clone_model(self.actor)
            self.actor_._name = "actor_"
            self.critic = self.default_build_critic_callback(critic_encoder, action_num)
            self.critic._name = "critic"
            self.critic_ = keras.models.clone_model(self.critic)
            self.critic_._name = "critic_"

    def predict(self, s) -> np.ndarray:
        return self.actor.predict(np.expand_dims(s, axis=0), verbose=0)[0]

    def save(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.actor.save_weights(os.path.join(model_tmp_dir, "actor.ckpt"))
        self.critic.save_weights(os.path.join(model_tmp_dir, "critic.ckpt"))
        tools.zip_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if self.actor is None:
            raise TypeError("network has not been build yet")
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.actor.load_weights(os.path.join(unzipped_dir, "actor.ckpt"))
        if self.training:
            self.actor_.load_weights(os.path.join(unzipped_dir, "actor.ckpt"))
            self.critic.load_weights(os.path.join(unzipped_dir, "critic.ckpt"))
            self.critic_.load_weights(os.path.join(unzipped_dir, "critic.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)
