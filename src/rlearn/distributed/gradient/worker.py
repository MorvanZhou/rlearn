import json
import logging
import os
import random
import string
import tempfile
import typing as tp
import zlib
from uuid import uuid4

import grpc
import numpy as np

from rlearn.distributed import tools
from rlearn.distributed.gradient import param_pb2, param_pb2_grpc
from rlearn.distributed.logger import get_logger
from rlearn.env.env_wrapper import EnvWrapper
from rlearn.trainer import get_trainer_by_name
from rlearn.trainer.base import BaseTrainer
from rlearn.trainer.tools import set_trainer_action_transformer


def connect_params_server(
        logger: logging.Logger,
        params_server_address: str,
        timeout: int = 15,
) -> tp.Tuple[param_pb2_grpc.ParamsStub, grpc.Channel]:
    channel = grpc.insecure_channel(params_server_address)
    stub = param_pb2_grpc.ParamsStub(channel=channel)
    # get setting
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise ValueError(f"connect params server at {params_server_address} timeout: {timeout}")

    logger.debug("params server connected")
    return stub, channel


def iter_gradients(gradients: np.ndarray, chunk_size: int = 1024):
    yield param_pb2.SyncReq(requestId=str(uuid4()))
    b_weights = gradients.astype(np.float32).tobytes()
    compressed_data = zlib.compress(b_weights)
    for i in range(0, len(compressed_data), chunk_size):
        yield param_pb2.SyncReq(chunkData=compressed_data[i: i + chunk_size])


def parse_weights(logger: logging.Logger, resp_iter) -> tp.Tuple[np.ndarray, bool]:
    data = bytearray()
    stop = False
    for r in resp_iter:
        if r.chunkData == b"":
            stop = r.meta.stop
            logger.debug(
                """ReplicateModel | {"requestId": '%s', "stop": %s}""",
                r.meta.requestId, r.meta.stop)
            continue
        data.extend(r.chunkData)
    d_data = zlib.decompress(data)
    weights = np.frombuffer(d_data, dtype=np.float32)
    return weights, stop


def init(logger: logging.Logger, stub: param_pb2_grpc.ParamsStub) -> tp.Tuple[BaseTrainer, int, int]:
    resp_iter = stub.Start(param_pb2.StartReq(
        requestId=str(uuid4())
    ))
    data = bytearray()
    trainer: tp.Optional[BaseTrainer] = None
    max_episode_step = 0
    sync_step = 5
    filepath = ""
    for r in resp_iter:
        if r.chunkData == b"":
            filepath = os.path.normpath(os.path.join(tempfile.gettempdir(), logger.name, "param_server_pb.zip"))
            trainer = get_trainer_by_name(r.meta.trainerType)
            trainer.set_params(
                batch_size=r.meta.batchSize,
                gamma=r.meta.gamma,
                min_epsilon=0.1,
                epsilon_decay=1e-3,
            )
            trainer.set_replay_buffer(max_size=r.meta.bufferSize, buf=r.meta.bufferType)
            if r.meta.actionTransform != "":
                try:
                    at = json.loads(r.meta.actionTransform)
                except json.JSONDecodeError:
                    at = None
                set_trainer_action_transformer(trainer, at)
            max_episode_step = r.meta.maxEpisodeStep
            sync_step = r.meta.syncStep
            continue
        data.extend(r.chunkData)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(data)

    trainer.load_model(filepath)
    logger.debug("initial model is set from path='%s'", filepath)
    os.remove(filepath)
    return trainer, max_episode_step, sync_step


def sync(logger: logging.Logger, stub: param_pb2_grpc.ParamsStub, trainer: BaseTrainer) -> bool:
    gradients = trainer.compute_flat_gradients()
    if gradients is None:
        return False

    resp_iter = stub.Sync(iter_gradients(
        gradients=gradients, chunk_size=1024
    ))
    weights, stop = parse_weights(logger=logger, resp_iter=resp_iter)
    trainer.model.set_flat_weights(weights=weights)
    return stop


def run(
        env: EnvWrapper,
        params_server_address: str,
        name: str = "",
        debug: bool = False
):
    if name == "":
        name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
    logger = get_logger(f"worker-{name}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    stub, channel = connect_params_server(
        logger=logger, params_server_address=params_server_address, timeout=15)

    trainer, max_episode_step, sync_step = init(logger=logger, stub=stub)

    ep = 0
    stop = False
    while True:
        ep_r = 0
        s = env.reset()

        step = 0
        episode_step_generator = tools.get_count_generator(max_episode_step)
        for step in episode_step_generator:

            _a = trainer.predict(s)
            a = trainer.map_action(_a)

            s_, r, done = env.step(a)
            trainer.store_transition(s=s, a=_a, r=r, s_=s_, done=done)
            s = s_
            ep_r += r
            if done or (step % sync_step == 0 and step != 0) or step == max_episode_step - 1:
                stop = sync(logger=logger, stub=stub, trainer=trainer)
            if done or stop:
                break

        logger.info("ep=%d, total_step=%d, ep_r=%.3f", ep, step, ep_r)
        if stop:
            break
        ep += 1

    channel.close()
