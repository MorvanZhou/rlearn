import json
import os
import socket
import tempfile
import typing as tp
import uuid
import zlib

import grpc
import numpy as np

from rlearn.distributed.experience import buffer_pb2, actor_pb2


class DataMeta(tp.Protocol):
    version: int
    requestId: str
    attributes: str


class DataInterface(tp.Protocol):
    meta: DataMeta
    chunkData: bytes


def unpack_downloaded_transitions(
        resp_iter: buffer_pb2.DownloadDataResp
) -> tp.Tuple[int, tp.Dict[str, np.ndarray], str, str]:
    data = bytearray()
    err = ""
    attributes = ""
    request_id = ""
    for req in resp_iter:
        if req.meta.attributes != "":
            request_id = req.meta.requestId
            err = req.meta.err
            attributes = req.meta.attributes
            continue
        data.extend(req.chunkData)
    batch_size, batch = _unpack_flat_transitions(attributes_str=attributes, data_bytes=data)
    return batch_size, batch, err, request_id


def unpack_uploaded_transitions(
        request_iterator: buffer_pb2.UploadDataReq,
) -> tp.Tuple[int, tp.Dict[str, np.ndarray], int, str]:
    data = bytearray()
    attributes = ""
    request_id = ""
    version = 0
    for req in request_iterator:
        if req.meta.attributes != "":
            request_id = req.meta.requestId
            attributes = req.meta.attributes
            version = req.meta.version
            continue
        data.extend(req.chunkData)
    batch_size, batch = _unpack_flat_transitions(attributes_str=attributes, data_bytes=data)
    return batch_size, batch, version, request_id


def _unpack_flat_transitions(attributes_str: str, data_bytes: bytes) -> tp.Tuple[int, tp.Dict[str, np.ndarray]]:
    try:
        attr = json.loads(attributes_str)
    except json.JSONDecodeError:
        return 0, {}
    d_data = zlib.decompress(data_bytes)
    data = np.frombuffer(d_data, dtype=np.float32)
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


def pack_transitions_for_uploading(
        buffer, version: int, max_size: int = None, chunk_size: int = 1024, request_id: str = None
) -> buffer_pb2.UploadDataReq:
    data, keys = _get_data_and_keys(buffer=buffer, max_size=max_size)
    req = buffer_pb2.UploadDataReq(
        meta=buffer_pb2.UploadDataMeta(
            attributes=json.dumps([{"name": k, "shape": data[k].shape} for k in keys]),
            version=version,
            requestId=request_id if request_id is not None else str(uuid.uuid4())
        )
    )
    yield req

    v = np.concatenate([data[k].ravel() for k in keys], dtype=np.float32)
    v_bytes = v.tobytes()
    v_compressed = zlib.compress(v_bytes)
    for i in range(0, len(v_compressed), chunk_size):
        req = buffer_pb2.UploadDataReq(chunkData=v_compressed[i: i + chunk_size])
        yield req


def pack_transitions_for_downloading(
        buffer, max_size: int, request_id: str, chunk_size: int = 1024,
) -> buffer_pb2.DownloadDataResp:
    if buffer.is_empty():
        yield buffer_pb2.DownloadDataResp(
            meta=buffer_pb2.DownloadDataMeta(
                attributes="",
                err="no data",
                requestId=request_id
            )
        )
        return

    data, keys = _get_data_and_keys(buffer=buffer, max_size=max_size)

    resp = buffer_pb2.DownloadDataResp(
        meta=buffer_pb2.DownloadDataMeta(
            attributes=json.dumps([{"name": k, "shape": data[k].shape} for k in keys]),
            err="",
            requestId=request_id
        )
    )
    yield resp

    v = np.concatenate([data[k].ravel() for k in keys], dtype=np.float32)
    v_bytes = v.tobytes()
    v_compressed = zlib.compress(v_bytes)
    for i in range(0, len(v_compressed), chunk_size):
        resp = buffer_pb2.DownloadDataResp(chunkData=v_compressed[i: i + chunk_size])
        yield resp
    buffer.clear()


def _get_data_and_keys(buffer, max_size):
    if max_size is None or max_size > buffer.current_loading_point:
        data = buffer.get_current_loading()
    else:
        data = {}
        for k, v in buffer.data.items():
            data[k] = v[:max_size]

    keys = list(data.keys())
    keys.sort()
    return data, keys


def read_pb_iterfile(
        filepath: str,
        trainer_type: str,
        buffer_size: int,
        buffer_type: str,
        max_episode: int,
        max_episode_step: int,
        action_transform: list,
        version: int,
        chunk_size=1024,
        request_id: tp.Optional[str] = None,
) -> actor_pb2.StartReq:
    if request_id is None:
        request_id = str(uuid.uuid4())

    yield actor_pb2.StartReq(meta=actor_pb2.StartMeta(
        filename=os.path.basename(filepath),
        trainerType=trainer_type,
        bufferSize=buffer_size,
        bufferType=buffer_type,
        maxEpisode=max_episode,
        version=version,
        maxEpisodeStep=max_episode_step,
        actionTransform=json.dumps(action_transform),
        requestId=request_id
    ))

    with open(filepath, mode="rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                entry_request = actor_pb2.StartReq(chunkData=chunk)
                yield entry_request
            else:  # The chunk was empty, which means we're at the end of the file
                return


# def read_weights_iterfile(filepath, version: int, chunk_size=1024, request_id: str = None):
#     if request_id is None:
#         request_id = str(uuid.uuid4())
#     yield actor_pb2.ReplicateModelReq(meta=actor_pb2.ModelMeta(
#         filename=os.path.basename(filepath),
#         version=version,
#         requestId=request_id
#     ))
#     with open(filepath, mode="rb") as f:
#         while True:
#             chunk = f.read(chunk_size)
#             if chunk:
#                 entry_request = actor_pb2.ReplicateModelReq(chunkData=chunk)
#                 yield entry_request
#             else:  # The chunk was empty, which means we're at the end of the file
#                 return


def get_iter_values(
        msg_handler,
        values: np.ndarray,
        version: int,
        chunk_size: int = 1024,
        request_id: str = None,
):
    if request_id is None:
        request_id = str(uuid.uuid4())
    yield msg_handler(meta=actor_pb2.ModelMeta(
        version=version,
        requestId=request_id
    ))
    b_weights = values.astype(np.float32).tobytes()
    compressed_data = zlib.compress(b_weights)
    for i in range(0, len(compressed_data), chunk_size):
        yield msg_handler(chunkData=compressed_data[i: i + chunk_size])


def parse_resp_to_flat_gradients(resp_iter) -> tp.Tuple[np.ndarray, int, str]:
    data = bytearray()
    version = 0
    request_id = ""
    for resp in resp_iter:
        if resp.chunkData == b"":
            version = resp.meta.version
            request_id = resp.meta.requestId
            continue
        data.extend(resp.chunkData)
    d_data = zlib.decompress(data)
    gradients = np.frombuffer(d_data, dtype=np.float32)
    return gradients, version, request_id


def replicate_model(request_iterator, logger, model_loaded_event, weights_conn):
    data = bytearray()
    version = 0
    req_id = ""

    for req in request_iterator:
        if req.chunkData == b"":
            version = req.meta.version
            req_id = req.meta.requestId
            logger.debug(
                """ReplicateModel | {"requestId": '%s', "version": %d}""",
                req.meta.requestId, req.meta.version)
            continue
        data.extend(req.chunkData)
    d_data = zlib.decompress(data)
    weights = np.frombuffer(d_data, dtype=np.float32)

    model_loaded_event.clear()
    weights_conn.send([version, weights])
    if model_loaded_event.wait(5):
        done = True
        err = ""
    else:
        done = False
        err = "model replicating timeout"
    return actor_pb2.ReplicateModelResp(done=done, err=err, requestId=req_id)


def initialize_model(request_iterator, logger, process, resp):
    data = bytearray()
    filepath = ""
    request_id = ""
    tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    os.makedirs(tmp_dir)
    for req in request_iterator:
        if req.meta.filename != "":
            logger.debug(
                'Start | '
                '{"reqId": "%s", "trainerType": "%s","filename": "%s", "version": %d, "bufferType": %s, '
                '"bufferSize": %d, maxEpisode": %d, "maxEpisodeStep": %d, "actionTransform": "%s"}',
                req.meta.requestId,
                req.meta.trainerType,
                req.meta.filename,
                req.meta.version,
                req.meta.bufferType,
                req.meta.bufferSize,
                req.meta.maxEpisode,
                req.meta.maxEpisodeStep,
                req.meta.actionTransform,
            )

            filepath = os.path.normpath(os.path.join(tmp_dir, req.meta.filename))
            request_id = req.meta.requestId
            try:
                at = json.loads(req.meta.actionTransform)
            except json.JSONDecodeError:
                at = None
            else:
                if len(at) == 0:
                    at = None

            process.init_params(
                trainer_type=req.meta.trainerType,
                model_pb_path=filepath,
                buffer_type=req.meta.bufferType,
                buffer_size=req.meta.bufferSize,
                init_version=req.meta.version,
                request_id=request_id,
                max_episode=req.meta.maxEpisode,
                max_episode_step=req.meta.maxEpisodeStep,
                action_transform=at,
            )

            continue
        data.extend(req.chunkData)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(data)

    process.start()
    timeout = 15
    if process.model_loaded.wait(timeout):
        done = True
        err = ""
    else:
        done = False
        err = f"model load timeout ({timeout}s)"
    return resp(done=done, err=err, requestId=request_id)


def learner_init_actors(self):
    path = os.path.join(tempfile.gettempdir(), f"params_v{self.version_count}.zip")
    self.trainer.save_model(path)
    futures = {}
    if self.trainer.model.action_transformer is None:
        at = []
    else:
        at = self.trainer.model.action_transformer.params

    for addr, stub in self.actors_stub.items():
        req_id = str(uuid.uuid4())
        try:
            resp_future = stub.Start.future(
                read_pb_iterfile(
                    filepath=path,
                    trainer_type=self.trainer.name,
                    buffer_type=self.actor_buffer_type,
                    buffer_size=self.actor_buffer_size,
                    max_episode=0,
                    max_episode_step=0,
                    action_transform=at,
                    version=self.version_count,
                    request_id=req_id,
                )
            )
            futures[addr] = resp_future
        except grpc.RpcError as e:
            self.logger.error("actor start err: %e, addr='%s', requestId='%s'", str(e), addr, req_id)

    for addr, resp_future in futures.items():
        resp = resp_future.result()
        if not resp.done:
            raise ValueError(f"actor start err: {resp.err}, addr='{addr}', requestId='{resp.requestId}'")

    self.logger.debug("init actors with version=%d", self.version_count)
    self.version_count += 1


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
