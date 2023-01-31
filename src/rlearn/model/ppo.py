from abc import ABC

from tensorflow import keras

from rlearn.model.base import BaseStochasticModel


class _PPO(BaseStochasticModel, ABC):
    is_on_policy = True
    predicted_model_name = "pi_"

    def __init__(self, training: bool = True, ):
        super().__init__(training=training)

    @staticmethod
    def set_critic_encoder_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def set_encoder(self, pi: keras.Model, critic: keras.Model, action_num: int):
        old_pi = self.set_actor_encoder_callback(pi, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic)
        self.set_model(old_pi, c)

    def set_model(self, pi: keras.Model, critic: keras.Model):
        self.models["pi_"] = pi
        if self.training:
            self.models["pi"] = self.clone_model(self.models["pi_"])
            self.models["critic"] = critic

    @staticmethod
    def build_critic_callback(encoder: keras.Sequential):
        o = keras.layers.Dense(1)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])


class PPODiscrete(_PPO):
    name = __qualname__
    is_discrete_action = True

    def __init__(self, training: bool = True, ):
        super().__init__(training=training)


class PPOContinue(_PPO):
    name = __qualname__
    is_discrete_action = False

    def __init__(self, training: bool = True, ):
        super().__init__(training=training)
