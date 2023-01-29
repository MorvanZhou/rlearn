import tensorflow as tf
from tensorflow import keras

from rlearn.model.base import BaseStochasticModel


class _SAC(BaseStochasticModel):
    name = __qualname__

    def __init__(
            self,
            is_discrete: bool,
            training: bool = True,
    ):
        super().__init__(is_discrete=is_discrete, training=training)
        self.predicted_model_name = "actor"

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
            self.models["critic"] = critic
            self.models["critic_"] = self.clone_model(self.models["critic"])


class SACDiscrete(_SAC):
    name = __qualname__

    def __init__(self, training: bool = True):
        super().__init__(is_discrete=True, training=training)


class SACContinue(_SAC):
    name = __qualname__

    def __init__(self, training: bool = True, ):
        super().__init__(is_discrete=False, training=training)
