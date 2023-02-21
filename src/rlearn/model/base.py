import json
import os
import shutil
import typing as tp
import zipfile
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

import rlearn.type
from rlearn.transformer import BaseTransformer


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
        if os.path.exists(dest_dir):
            print(f"folder exist at {dest_dir}, use cached files")
            return dest_dir
        os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    return dest_dir


class BaseRLModel(ABC):
    def __init__(
            self,
            training: bool = True,
    ):
        self.training: bool = training
        self.models: tp.Dict[str, keras.Model] = {}
        self.action_transformer: tp.Optional[BaseTransformer] = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        ...

    @classmethod
    @property
    @abstractmethod
    def is_discrete_action(cls):
        ...

    @classmethod
    @property
    @abstractmethod
    def predicted_model_name(cls):
        ...

    @classmethod
    @property
    @abstractmethod
    def is_on_policy(cls):
        ...

    @abstractmethod
    def set_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray):
        pass

    @abstractmethod
    def disturbed_action(self, x, epsilon: float):
        pass

    def set_action_transformer(self, transformer: BaseTransformer):
        self.action_transformer = transformer

    def map_action(self, action: rlearn.type.Action) -> rlearn.type.Action:
        if self.action_transformer is None:
            return action
        return self.action_transformer.transform(action)

    def mapped_predict(self, x: np.ndarray) -> rlearn.type.Action:
        a = self.predict(x)
        return self.map_action(a)

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
        self.get_model_for_prediction().load_weights(
            os.path.join(unzipped_dir, f"{self.predicted_model_name}.ckpt"))
        if self.training:
            for k, v in self.models.items():
                if k == self.predicted_model_name:
                    continue
                v.load_weights(os.path.join(unzipped_dir, f"{k}.ckpt"))
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    def get_flat_weights(self) -> np.ndarray:

        weights = []
        keys = list(self.models.keys())
        keys.sort()
        for model_name in keys:
            model = self.models[model_name]
            if not self.training and model_name != self.predicted_model_name:
                continue
            for layer in model.layers:
                if len(layer.weights) == 0:
                    continue
                layer_shape = []
                for w in layer.get_weights():
                    layer_shape.append(w.shape)
                    weights.append(w.ravel())
        return np.concatenate(weights, dtype=np.float32, axis=0)

    def set_flat_weights(self, weights: np.ndarray):
        assert weights.dtype == np.float32, TypeError(f"gradients must be np.float32, but got {weights.dtype}")
        p = 0
        keys = list(self.models.keys())
        keys.sort()
        for model_name in keys:
            model = self.models[model_name]
            if not self.training and model_name != self.predicted_model_name:
                continue
            for layer in model.layers:
                if len(layer.weights) == 0:
                    continue
                layer_weights = []
                for lw in layer.weights:
                    p_ = p + np.prod(lw.shape)
                    w = weights[p: p_]
                    layer_weights.append(w.reshape(lw.shape))
                    p = p_
                layer.set_weights(layer_weights)
        assert p == len(weights), ValueError("reshape size not right")

    def save(self, path: str):
        model_tmp_dir = path
        if path.endswith(".zip"):
            model_tmp_dir = path.rsplit(".zip")[0]
        for k, v in self.models.items():
            keras.models.save_model(v, os.path.join(model_tmp_dir, k), include_optimizer=False)
        with open(os.path.join(model_tmp_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump({
                "modelName": self.name,
            },
                f, indent=2, ensure_ascii=False)
        zip_pb_model(model_tmp_dir)
        shutil.rmtree(model_tmp_dir, ignore_errors=True)

    def load(self, path: str):
        if not path.endswith(".zip"):
            path += ".zip"
        unzipped_dir = unzip_model(path)
        self.models[self.predicted_model_name] = keras.models.load_model(
            os.path.join(unzipped_dir, self.predicted_model_name),
            compile=False,
        )
        if self.training:
            for filename in os.listdir(unzipped_dir):
                if filename.endswith(".json"):
                    continue
                if os.path.isfile(os.path.join(unzipped_dir, filename)):
                    continue
                model_name = filename.rsplit(".zip", 1)[0]
                self.models[model_name] = keras.models.load_model(
                    os.path.join(unzipped_dir, model_name),
                    compile=False
                )
        shutil.rmtree(unzipped_dir, ignore_errors=True)

    def get_model_for_prediction(self) -> keras.Model:
        return self.models[self.predicted_model_name]

    @staticmethod
    def clone_model(model):
        try:
            new_model = keras.models.clone_model(model)
        except ValueError:
            new_model = type(model)()
            new_model.set_weights(model.get_weights())
        return new_model


class BaseStochasticModel(BaseRLModel, ABC):
    def __init__(self, training: bool = True):
        super().__init__(training=training)

    def dist(self, net: keras.Model, s: np.ndarray):
        if self.is_discrete_action:
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
        if self.is_discrete_action:
            o = keras.layers.Dense(action_num)(encoder.output)
            return keras.Model(inputs=encoder.inputs, outputs=[o])

        o = keras.layers.Dense(action_num * 2)(encoder.output)
        return keras.Model(inputs=encoder.inputs, outputs=[o])
