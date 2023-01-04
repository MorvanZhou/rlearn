from tensorflow import keras

from rlearn.model.dqn import DQN


class DuelingDQN(DQN):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    @staticmethod
    def default_build_callback(encoder: keras.Sequential, action_num: int):
        oa = keras.layers.Dense(64)(encoder.output)
        oa = keras.layers.Dense(action_num)(oa)
        ov = keras.layers.Dense(64)(encoder.output)
        ov = keras.layers.Dense(1)(ov)
        q = ov + oa
        return keras.Model(inputs=encoder.inputs, outputs=[q])
