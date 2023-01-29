import os
import shutil
import typing as tp
import zipfile
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras


def zip_ckpt_model(src_dir, dest_path=None):
    if dest_path is None:
        dest_path = src_dir + ".zip"
    dest_dir = os.path.dirname(dest_path)
    if dest_dir != "":
        os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(dest_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(src_dir):
            if ".ckpt." not in filename:
                continue
            filepath = os.path.join(src_dir, filename)
            zipf.write(filepath, os.path.relpath(filepath, src_dir))


def zip_pb_model(src_dir, dest_path=None):
    if dest_path is None:
        dest_path = src_dir + ".zip"
    dest_dir = os.path.dirname(dest_path)
    if dest_dir != "":
        os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(dest_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, filenames in os.walk(src_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                zipf.write(filepath, os.path.relpath(filepath, src_dir))


def unzip_model(path, dest_dir=None):
    if not path.endswith(".zip"):
        path += ".zip"
    if dest_dir is None:
        dest_dir = os.path.normpath(path).rsplit(".zip")[0]
        os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    return dest_dir


class BaseRLModel(ABC):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        self.training = training
        self.models: tp.Dict[str, keras.Model] = {}
        self.predicted_model_name = ""

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def set_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def disturbed_action(self, x, epsilon: float):
        pass

    def save_weights(self, path):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        for k, v in self.models.items():
            v.save_weights(os.path.join(model_tmp_dir, f"{k}.ckpt"))
        zip_ckpt_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load_weights(self, path):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = unzip_model(path)
        self.models[self.predicted_model_name].load_weights(
            os.path.join(unzipped_dir, f"{self.predicted_model_name}.ckpt"))
        if self.training:
            for k, v in self.models.items():
                if k == self.predicted_model_name:
                    continue
                v.load_weights(os.path.join(unzipped_dir, f"{k}.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    def save(self, path: str):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        for k, v in self.models.items():
            v.save(os.path.join(model_tmp_dir, k))
        zip_pb_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load(self, path: str):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = unzip_model(path)
        self.models[self.predicted_model_name] = keras.models.load_model(
            os.path.join(unzipped_dir, self.predicted_model_name))
        if self.training:
            for filename in os.listdir(unzipped_dir):
                model_name = filename.rsplit(".zip", 1)[0]
                self.models[model_name] = keras.models.load_model(os.path.join(unzipped_dir, model_name))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    @staticmethod
    def clone_model(model):
        try:
            new_model = keras.models.clone_model(model)
        except ValueError:
            new_model = type(model)()
            new_model.set_weights(model.get_weights())
        return new_model


class BaseStochasticModel(BaseRLModel, ABC):
    def __init__(self, is_discrete: bool, training: bool = True):
        super().__init__(training=training)
        self.is_discrete = is_discrete

    def dist(self, net: keras.Model, s: np.ndarray):
        if self.is_discrete:
            o = net(s)
            return tfp.distributions.Categorical(logits=o)

        o = net(s)
        a_size = o.shape[1] // 2
        loc, scale = tf.tanh(o[:, :a_size]), tf.nn.sigmoid(o[:, a_size:])
        return tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    def predict(self, s: np.ndarray):
        s = np.expand_dims(s, axis=0)
        dist = self.dist(self.models[self.predicted_model_name], s)  # use stable policy parameters to predict
        action = tf.squeeze(dist.sample(1), axis=[0, 1]).numpy()
        if np.isnan(action).any():
            raise ValueError("action contains NaN")
        if action.ndim == 0 and np.issubdtype(action, np.integer):
            action = int(action)
        return action

    def disturbed_action(self, x, epsilon: float):
        return self.predict(x)

    def set_actor_encoder_callback(self, encoder: keras.Sequential, action_num: int) -> keras.Model:
        if self.is_discrete:
            o = keras.layers.Dense(action_num)(encoder.output)
            return keras.Model(inputs=encoder.inputs, outputs=[o])

        o = keras.layers.Dense(action_num * 2)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])
