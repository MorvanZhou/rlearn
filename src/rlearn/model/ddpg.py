import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLModel


class DDPG(BaseRLModel):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        BaseRLModel.__init__(self, training=training)
        self.critic = None
        self.critic_ = None
        self.actor = None
        self.actor_ = None

    @staticmethod
    def set_actor_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num, activation="tanh")(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential, action_num: int):
        action_inputs = keras.layers.InputLayer(
            input_shape=(action_num,),
            name="action_inputs",
            dtype=tf.float32)
        encoding = tf.concat([encoder.output, action_inputs.output], axis=1)
        o = keras.layers.Dense(action_num * 16)(encoding)
        o = keras.layers.ReLU()(o)
        o = keras.layers.Dense(1)(o)
        return keras.Model(inputs=encoder.inputs + [action_inputs.input, ], outputs=[o])

    def set_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        a = self.set_actor_encoder_callback(actor, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic, action_num)
        self.set_model(a, c)

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.actor = actor
        if self.training:
            self.actor_ = self.clone_model(self.actor)
            self.critic = critic
            self.critic_ = self.clone_model(self.critic)

    def predict(self, s) -> np.ndarray:
        return self.actor.predict(np.expand_dims(s, axis=0), verbose=0)[0]

    def save_weights(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.actor.save_weights(os.path.join(model_tmp_dir, "actor.ckpt"))
        self.critic.save_weights(os.path.join(model_tmp_dir, "critic.ckpt"))
        tools.zip_ckpt_model(model_tmp_dir)
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
            self.actor_ = keras.models.load_model(os.path.join(unzipped_dir, "actor"))
            self.critic = keras.models.load_model(os.path.join(unzipped_dir, "critic"))
            self.critic_ = keras.models.load_model(os.path.join(unzipped_dir, "critic"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)
