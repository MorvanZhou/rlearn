import json
import os
import socket
import typing as tp
import uuid

import numpy as np

from rlearn.distribute.experience import actor_pb2 as exp_actor_pb2


class PackedData(tp.Protocol):
    values: tp.List[float]
    attributes: str


class DataInterface(tp.Protocol):
    data: PackedData
    requestId: str


def unpack_transitions(interface: DataInterface) -> tp.Tuple[int, tp.Dict[str, np.ndarray]]:
    data, attributes = interface.data.values[:], interface.data.attributes
    try:
        attr = json.loads(attributes)
    except json.JSONDecodeError:
        return 0, {}

    batch = {}
    p = 0
    batch_size = None
    for attr_shape in attr:
        name = attr_shape["name"]
        shape = attr_shape["shape"]
        p_ = np.prod(shape) + p
        b = np.reshape(data[p:p_], newshape=shape)
        p = p_
        batch[name] = b
        if batch_size is None:
            batch_size = b.shape[0]
    if batch_size is None:
        batch_size = 0
    return batch_size, batch


def pack_transitions(buffer, interface: DataInterface, max_size: int = None):
    if max_size is None or max_size > buffer.current_loading_point:
        data = buffer.get_current_loading()
    else:
        data = {}
        for k, v in buffer.data.items():
            data[k] = v[:max_size]

    keys = list(data.keys())
    v = np.concatenate([data[k].ravel() for k in keys])
    interface.data.values[:] = v
    interface.data.attributes = json.dumps([{"name": k, "shape": data[k].shape} for k in keys])
    if interface.requestId == "":
        interface.requestId = str(uuid.uuid4())
    return interface


def read_pb_iterfile(
        filepath: str,
        trainer_type: str,
        max_episode: int,
        max_episode_step: int,
        version: int,
        chunk_size=1024,
        request_id: tp.Optional[str] = None,
) -> exp_actor_pb2.StartReq:
    if request_id is None:
        request_id = str(uuid.uuid4())

    yield exp_actor_pb2.StartReq(meta=exp_actor_pb2.StartMeta(
        filename=os.path.basename(filepath),
        trainerType=trainer_type,
        maxEpisode=max_episode,
        version=version,
        maxEpisodeStep=max_episode_step,
        requestId=request_id
    ))

    with open(filepath, mode="rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                entry_request = exp_actor_pb2.StartReq(chunkData=chunk)
                yield entry_request
            else:  # The chunk was empty, which means we're at the end of the file
                return


def read_weights_iterfile(filepath, version: int, chunk_size=1024, request_id: str = None):
    if request_id is None:
        request_id = str(uuid.uuid4())
    yield exp_actor_pb2.ReplicateModelReq(meta=exp_actor_pb2.ReplicateModelMeta(
        filename=os.path.basename(filepath),
        version=version,
        requestId=request_id
    ))
    with open(filepath, mode="rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                entry_request = exp_actor_pb2.ReplicateModelReq(chunkData=chunk)
                yield entry_request
            else:  # The chunk was empty, which means we're at the end of the file
                return


def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_count_generator(max_count):
    def unlimited_count_generator():
        c = 0
        while True:
            yield c
            c += 1

    def fix_count_generator(max_step: int):
        for c in range(max_step):
            yield c

    if max_count <= 0:
        g = unlimited_count_generator()
    else:
        g = fix_count_generator(max_count)
    return g
