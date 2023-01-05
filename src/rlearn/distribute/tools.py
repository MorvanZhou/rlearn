import json
import os
import socket
import typing as tp
import uuid

import numpy as np

from rlearn.distribute import actor_pb2


class PackedData(tp.Protocol):
    values: tp.List[float]
    attributes: str


class DataInterface(tp.Protocol):
    data: PackedData
    requestId: str


def unpack_transitions(interface: DataInterface):
    data, attributes = interface.data.values[:], interface.data.attributes
    attr = json.loads(attributes)
    p = np.prod(attr["s_shape"])
    s = np.reshape(data[:p], newshape=attr["s_shape"])
    p_ = p + np.prod(attr["a_shape"])
    a = np.reshape(data[p:p_], newshape=attr["a_shape"])
    p = p_
    r = np.reshape(data[p:], newshape=attr["r_shape"])
    s_ = None
    if attr["has_next_state"]:
        s_ = s[:, 1]
        s = s[:, 0]
    return s, a, r, s_


def pack_transitions(buffer, interface: DataInterface, max_size: int = None):
    if max_size is None or max_size > buffer.current_loading_point:
        s, a, r = buffer.get_current_loading()
    else:
        s = buffer.s[:max_size]
        a = buffer.a[:max_size]
        r = buffer.r[:max_size]

    v = np.concatenate([s.ravel(), a.ravel(), r.ravel()])
    interface.data.values[:] = v
    interface.data.attributes = json.dumps({
        "s_shape": s.shape,
        "a_shape": a.shape,
        "r_shape": r.shape,
        "has_next_state": buffer.has_next_state,
    })
    if interface.requestId == "":
        interface.requestId = str(uuid.uuid4())
    return interface


def read_iterfile(filepath, chunk_size=1024):
    yield actor_pb2.ReplicateModelReq(filename=os.path.basename(filepath))
    with open(filepath, mode="rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                entry_request = actor_pb2.ReplicateModelReq(chunkData=chunk)
                yield entry_request
            else:  # The chunk was empty, which means we're at the end of the file
                return


def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
