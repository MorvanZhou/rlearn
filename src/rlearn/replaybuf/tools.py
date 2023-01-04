import typing as tp

from rlearn.replaybuf.base import BaseReplayBuffer

__BUF_MAP: tp.Dict[str, tp.Type[BaseReplayBuffer]] = {}


def _set_buf_map(cls, m: dict):
    for subclass in cls.__subclasses__():
        if subclass.__module__ != BaseReplayBuffer.__module__:
            m[subclass.__name__] = subclass
        _set_buf_map(subclass, m)


def get_buffer_by_name(name: str, max_size: int) -> BaseReplayBuffer:
    if len(__BUF_MAP) == 0:
        _set_buf_map(BaseReplayBuffer, __BUF_MAP)
    return __BUF_MAP[name](max_size)


def get_all_buffers():
    if len(__BUF_MAP) == 0:
        _set_buf_map(BaseReplayBuffer, __BUF_MAP)
    return __BUF_MAP
