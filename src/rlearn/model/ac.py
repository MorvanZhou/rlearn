
from tensorflow import keras

from rlearn.model.base import BaseStochasticModel


class _ActorCritic(BaseStochasticModel):
    is_on_policy = True

    def __init__(self, is_discrete: bool, training: bool = True, ):
        super().__init__(is_discrete=is_discrete, training=training)
        self.predicted_model_name = "actor"

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
        self.models["actor"] = actor
        self.predicted_model_name = "actor"
        if self.training:
            self.models["critic"] = critic

    @staticmethod
    def build_critic_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class ActorCriticDiscrete(_ActorCritic):
    name = __qualname__

    def __init__(self, training: bool = True, ):
        super().__init__(is_discrete=True, training=training)


class ActorCriticContinue(_ActorCritic):
    name = __qualname__

    def __init__(self, training: bool = True, ):
        super().__init__(is_discrete=False, training=training)
