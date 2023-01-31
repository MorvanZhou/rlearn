
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlearn.model.base import BaseRLModel


class DDPG(BaseRLModel):
    name = __qualname__
    is_on_policy = False
    is_discrete_action = False
    predicted_model_name = "actor"

    def __init__(
            self,
            training: bool = True,
    ):
        BaseRLModel.__init__(self, training=training)

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
        o = keras.layers.Dense(32)(o)
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
        self.models["actor"] = actor
        if self.training:
            self.models["actor_"] = self.clone_model(self.models["actor"])
            self.models["critic"] = critic
            self.models["critic_"] = self.clone_model(self.models["critic"])

    def predict(self, s) -> np.ndarray:
        a = self.models[self.predicted_model_name].predict(np.expand_dims(s, axis=0), verbose=0).ravel()
        if np.isnan(a).any():
            raise ValueError("action contains NaN")
        return a

    def disturbed_action(self, x, epsilon: float):
        if np.random.random() < epsilon:
            return np.random.uniform(-1., 1., size=self.models["actor"].output_shape[1])
        return self.predict(x)
