import numpy as np
from tensorflow import keras

from rlearn.model.base import BaseRLModel


class DQN(BaseRLModel):
    name = __qualname__
    is_discrete_action = True
    is_on_policy = False
    predicted_model_name = "q"

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    @staticmethod
    def set_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def set_encoder(self, encoder: keras.Model, action_num: int):
        q = self.set_encoder_callback(encoder, action_num)
        self.set_model(q)

    def set_model(self, q: keras.Model):
        self.models["q"] = q
        if self.training:
            self.models["q_"] = self.clone_model(self.models["q"])

    def predict(self, s: np.ndarray):
        """
        return: action index
        """
        s = np.expand_dims(s, axis=0)
        q = self.models[self.predicted_model_name].predict(s, verbose=0).ravel()
        if np.isnan(q).any():
            raise ValueError("action contains NaN")
        a_index = q.argmax()
        if a_index.ndim == 0 and np.issubdtype(a_index, np.integer):
            return int(a_index)
        return a_index

    def disturbed_action(self, x, epsilon: float):
        if np.random.random() < epsilon:
            a_size = self.models["q"].outputs[0].shape[1]
            return np.random.randint(0, a_size)
        return self.predict(x)
