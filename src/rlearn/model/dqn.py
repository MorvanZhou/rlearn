import os
import shutil
import typing as tp

import numpy as np
from tensorflow import keras

from rlearn.model import tools
from rlearn.model.base import BaseRLModel


class DQN(BaseRLModel):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)
        self.q: tp.Optional[keras.Model] = None
        self.q_: tp.Optional[keras.Model] = None

    @staticmethod
    def set_encoder_callback(encoder: keras.Sequential, action_num: int):
        o = keras.layers.Dense(action_num)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])

    def set_encoder(self, encoder: keras.Model, action_num: int):
        q = self.set_encoder_callback(encoder, action_num)
        self.set_model(q)

    def set_model(self, q: keras.Model):
        self.q = q
        if self.training:
            self.q_ = self.clone_model(self.q)

    def predict(self, s: np.ndarray):
        """
        return: action index
        """
        s = np.expand_dims(s, axis=0)
        q = self.q.predict(s, verbose=0).ravel()
        a_index = q.argmax()
        if a_index.ndim == 0 and np.issubdtype(a_index, np.integer):
            return int(a_index)
        return a_index

    def save_weights(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.q.save_weights(os.path.join(model_tmp_dir, "q.ckpt"))
        tools.zip_ckpt_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if self.q is None:
            raise TypeError("network has not been build yet")
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.q.load_weights(os.path.join(unzipped_dir, "q.ckpt"))
        if self.training:
            self.q_.load_weights(os.path.join(unzipped_dir, "q.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    def save(self, path: str):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        self.q.save(os.path.join(model_tmp_dir, "q"))
        tools.zip_pb_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load(self, path: str):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = tools.unzip_model(path)
        self.q = keras.models.load_model(os.path.join(unzipped_dir, "q"))
        if self.training:
            self.q_ = keras.models.load_model(os.path.join(unzipped_dir, "q"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)
