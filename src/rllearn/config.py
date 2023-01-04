import json
import os.path
import typing as tp
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum


class LayerType(Enum):
    relu = "relu"
    elu = "elu"
    leakyrelu = "leakyrelu"
    softmax = "softmax"


@dataclass
class LayerConfig:
    type: str
    args: tp.Dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class NetConfig:
    input_shape: tp.Sequence[int]
    layers: tp.Sequence[LayerConfig]


@dataclass
class ReplayBufferConfig:
    max_size: int
    buf: str = "RandomReplayBuffer"


@dataclass
class TrainConfig:
    trainer: str
    batch_size: int
    epochs: int
    nets: tp.Sequence[NetConfig]
    learning_rates: tp.Sequence[float]
    gamma: float
    action_transform: tp.List[tp.Any]
    replay_buffer: ReplayBufferConfig
    replace_step: int
    replace_ratio: float = 1.
    not_learn_epochs: int = -1
    min_epsilon: float = 0.1
    epsilon_decay: float = 1e-4
    args: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def dump(self, path: str):
        dump(self, path)


def load_json(path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return TrainConfig(
        trainer=js["trainer"],
        action_transform=js["action_transform"],
        min_epsilon=js["min_epsilon"],
        epsilon_decay=js["epsilon_decay"],
        gamma=js["gamma"],
        replay_buffer=js["buffer"],
        not_learn_epochs=js["not_learn_epochs"],
        replace_step=js["replace_step"],
        replace_ratio=js["replace_ratio"],
        epochs=js["epochs"],
        batch_size=js["batch_size"],
        nets=[NetConfig(
            input_shape=net["input_shape"],
            layers=[LayerConfig(type=layer["type"], args=layer["args"]) for layer in net["layers"]],
        ) for net in js["nets"]],
        learning_rates=js["learning_rates"],
        args=js["args"],
    )


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        elif isinstance(o, (set, tuple)):
            return list(o)
        return super().default(o)


def dump(conf: TrainConfig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json_str = json.dumps(conf, indent=2, ensure_ascii=False, cls=EnhancedJSONEncoder)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)






