import os
import typing as tp
import zipfile

from tensorflow import keras

from rlearn.config import NetConfig
from rlearn.model.base import BaseRLModel


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


LAYER_BUILD_MAP = {
    "relu": lambda args, trainable: keras.layers.ReLU(),
    "elu": lambda args, trainable: keras.layers.ELU(alpha=args.get("alpha", 1.0)),
    "leakyrelu": lambda args, trainable: keras.layers.LeakyReLU(alpha=args.get("alpha", 0.3)),
    "softmax": lambda args, trainable: keras.layers.Softmax(axis=args.get("axis", -1)),

    "conv2d": lambda args, trainable: keras.layers.Conv2D(
        filters=args["filters"],
        kernel_size=args["kernel_size"],
        strides=args.get("strides", (1, 1)),
        padding=args.get("padding", "valid"),
        data_format=None,
        activation=args.get("activation", None),
        use_bias=args.get("use_bias", True),
        trainable=trainable,
    ),
    "dense": lambda args, trainable: keras.layers.Dense(
        units=args["units"],
        activation=args.get("activation", None),
        use_bias=args.get("use_bias", True),
        trainable=trainable,
    ),
    "batchnorm": lambda args, trainable: keras.layers.BatchNormalization(
        axis=args.get("axis", -1),
        momentum=args.get("momentum", 0.99),
        epsilon=args.get("epsilon", 0.001),
        center=args.get("center", True),
        scale=args.get("scale", True),
        trainable=trainable
    ),

    "maxpool2d": lambda args, trainable: keras.layers.MaxPool2D(
        pool_size=args.get("pool_size", (2, 2)),
        strides=args.get("strides"),
        padding=args.get("padding", "valid")),
    "averagepooling2d": lambda args, trainable: keras.layers.AveragePooling2D(
        pool_size=args.get("pool_size", (2, 2)),
        strides=args.get("strides"),
        padding=args.get("padding", "valid")),
    "dropout": lambda args, trainable: keras.layers.Dropout(rate=args["rate"]),
    "flatten": lambda args, trainable: keras.layers.Flatten(),
    "reshape": lambda args, trainable: keras.layers.Reshape(target_shape=args["target_shape"]),
}


def build_net_from_config(
        net_config: NetConfig,
        action_num: int,
        callback: tp.Optional[tp.Callable[[keras.Model, int, str], keras.Model]] = None,
        trainable=True,
        name=None
) -> keras.Model:
    encoder = build_encoder_from_config(net_config, trainable)
    if callback is None:
        return keras.Model(inputs=encoder.inputs, outputs=encoder.outputs, name=name)
    return callback(encoder, action_num, name)


def build_encoder_from_config(
        net_config: NetConfig,
        trainable=True,
) -> keras.Sequential:
    layers = [keras.layers.InputLayer(input_shape=net_config.input_shape, name="inputs")]
    for layer in net_config.layers:
        layers.append(LAYER_BUILD_MAP[layer.type](layer.args, trainable))
    encoder = keras.Sequential(layers=layers)
    return encoder


__MODEL_MAP: tp.Dict[str, tp.Type[BaseRLModel]] = {}
__BASE_MODULE = BaseRLModel.__module__


def _set_model_map(cls, m: dict):
    for subclass in cls.__subclasses__():
        if subclass.__module__ != __BASE_MODULE and not subclass.__name__.startswith("_"):
            m[subclass.__name__] = subclass
        _set_model_map(subclass, m)


def get_model_by_name(
        name: str,
        training: bool = False,
) -> BaseRLModel:
    if len(__MODEL_MAP) == 0:
        _set_model_map(BaseRLModel, __MODEL_MAP)
    model = __MODEL_MAP[name](training=training)
    return model
