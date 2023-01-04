import os
import shutil
import typing as tp

import numpy as np
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLNet


class DQN(BaseRLNet):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)
        self.net: tp.Optional[keras.Model] = None
        self.net_: tp.Optional[keras.Model] = None

    @staticmethod
    def default_build_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def build(self, encoder: keras.Model, action_num: int):
        self.net = self.default_build_callback(encoder, action_num)
        self.net._name = "q_eval"
        if self.training:
            self.net_ = keras.models.clone_model(self.net)
            self.net_._name = "q_target"

    def predict(self, s: np.ndarray):
        """
        return: action index
        """
        s = np.expand_dims(s, axis=0)
        q = self.net.predict(s, verbose=0).ravel()
        a_index = q.argmax()
        if a_index.ndim == 0 and np.issubdtype(a_index, np.integer):
            return int(a_index)
        return a_index

    def save(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.net.save_weights(os.path.join(model_tmp_dir, "net.ckpt"))
        tools.zip_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if self.net is None:
            raise TypeError("network has not been build yet")
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.net.load_weights(os.path.join(unzipped_dir, "net.ckpt"))
        if self.training:
            self.net_.load_weights(os.path.join(unzipped_dir, "net.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)
