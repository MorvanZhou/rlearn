import json
import logging
import multiprocessing as mp
import os
import random
import string
import tempfile
import threading
import typing as tp
import uuid
from concurrent import futures

import grpc

from rlearn.distributed import tools
from rlearn.distributed.experience import actor_pb2, actor_pb2_grpc, buffer_pb2_grpc, buffer_pb2
from rlearn.distributed.logger import get_logger
from rlearn.env.env_wrapper import EnvWrapper
from rlearn.trainer.base import BaseTrainer
from rlearn.trainer.tools import get_trainer_by_name, set_trainer_action_transformer

# linus default is fork, force set to spawn
mp = mp.get_context('spawn')
_name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
logger = get_logger(f"actor-{_name}")


class ActorProcess(mp.Process):
    def __init__(
            self,
            local_buffer_size: int,
            env: EnvWrapper,
            remote_buffer_address: tp.Optional[str] = None,
            debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.trainer_type = ""
        self.init_model_pb_path = ""
        self.trainer: tp.Optional[BaseTrainer] = None
        self.local_buffer_size = local_buffer_size
        self.env: EnvWrapper = env
        self.max_episode = 0
        self.max_episode_step = 0
        self.buffer_address = remote_buffer_address
        self.training_request_id = ""
        self.action_transform = None

        mgr = mp.Manager()
        self.ns = mgr.Namespace()

        self.ns.new_model_path = ""
        self.ns.episode_num = 0
        self.ns.episode_step_num = 0
        self.ns.version = 0
        self.lock = mgr.Lock()
        self.model_loaded = mgr.Event()
        self.model_loaded.clear()

    def init_params(
            self,
            trainer_type: str,
            model_pb_path: str,
            init_version: int,
            request_id: str,
            max_episode: int = 0,
            max_episode_step: int = 0,
            action_transform: list = None,
    ):
        self.trainer_type = trainer_type
        self.init_model_pb_path = model_pb_path
        self.training_request_id = request_id
        if action_transform is not None:
            self.action_transform = action_transform
        self.ns.version = init_version
        self.max_episode = 0 if max_episode is None else max_episode
        self.max_episode_step = 0 if max_episode_step is None else max_episode_step

    def set_model(self):
        self.trainer = get_trainer_by_name(self.trainer_type)
        self.trainer.set_params(min_epsilon=0.1, epsilon_decay=1e-3)
        self.trainer.set_replay_buffer(max_size=self.local_buffer_size)
        if self.action_transform is not None:
            set_trainer_action_transformer(self.trainer, self.action_transform)

        self.trainer.load_model(self.init_model_pb_path)
        logger.debug("initial model is set from path='%s'", self.init_model_pb_path)
        os.remove(self.init_model_pb_path)
        self.model_loaded.set()

    def set_model_replicate(self, version: int, weights_path: str):
        with self.lock:
            self.ns.new_model_path = weights_path
            self.ns.version = version
            self.model_loaded.clear()

    def try_replicate_model(self):
        with self.lock:
            if self.ns.new_model_path != "":
                self.trainer.load_model_weights(self.ns.new_model_path)
                os.remove(self.ns.new_model_path)
                self.ns.new_model_path = ""
                self.model_loaded.set()
                logger.debug("model parameters replaced, version=%d", self.ns.version)

    def send_data_to_remote_buffer(self, buf_stub):
        req = buffer_pb2.UploadDataReq(version=self.ns.version, requestId=self.training_request_id)
        tools.pack_transitions(self.trainer.replay_buffer, req)
        resp = buf_stub.UploadData(req)
        if not resp.done:
            raise ValueError(f"grpc upload data to buffer err: {resp.err}")

    def run(self):
        logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        episode_generator = tools.get_count_generator(self.max_episode)

        if self.buffer_address is None or self.buffer_address.strip() == "":
            buf_stub = None
            channel = None
        else:
            channel = grpc.insecure_channel(self.buffer_address)
            buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        self.set_model()

        with self.lock:
            self.ns.episode_num = 0

        for ep in episode_generator:
            with self.lock:
                self.ns.episode_num = ep
            s = self.env.reset()

            step = 0
            episode_step_generator = tools.get_count_generator(self.max_episode_step)
            for step in episode_step_generator:
                with self.lock:
                    self.ns.episode_step_num = step

                self.try_replicate_model()

                _a = self.trainer.predict(s)
                a = self.trainer.map_action(_a)

                s_, r, done = self.env.step(a)
                self.trainer.store_transition(s=s, a=_a, r=r, s_=s_, done=done)
                s = s_
                if self.trainer.replay_buffer.is_full() and buf_stub is not None:
                    self.send_data_to_remote_buffer(buf_stub)
                    self.trainer.replay_buffer.clear()
                if done:
                    break

            logger.debug("ep=%d, total_step=%d", ep, step)

        if channel is not None:
            channel.close()


class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(self, actor: ActorProcess, stop_event: threading.Event, debug=False):
        self.actor = actor
        self.stop_event = stop_event

        logger.setLevel(logging.DEBUG if debug else logging.ERROR)
        with grpc.insecure_channel(self.actor.buffer_address) as channel:
            timeout = 15
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                raise ValueError(f"connect replay buffer at {self.actor.buffer_address} timeout: {timeout}")
        logger.debug("connected to buffer %s", self.actor.buffer_address)

    def ServiceReady(self, request, context):
        logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return actor_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request_iterator, context):
        data = bytearray()
        filepath = ""
        request_id = ""
        tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.makedirs(tmp_dir)
        for req in request_iterator:
            if req.meta.filename != "":
                logger.debug(
                    'Start | '
                    '{"reqId": "%s", "trainerType": "%s","filename": "%s", "version": %d, '
                    '"maxEpisode": %d, "maxEpisodeStep": %d, "actionTransform": "%s"}',
                    req.meta.requestId,
                    req.meta.trainerType,
                    req.meta.filename,
                    req.meta.version,
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

                self.actor.init_params(
                    trainer_type=req.meta.trainerType,
                    model_pb_path=filepath,
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

        self.actor.start()
        timeout = 15
        if self.actor.model_loaded.wait(timeout):
            done = True
            err = ""
        else:
            done = False
            err = f"model load timeout ({timeout}s)"
        return actor_pb2.StartResp(done=done, err=err, requestId=request_id)

    def ReplicateModel(self, request_iterator, context):
        data = bytearray()
        filepath = ""
        version = 0
        request_id = ""
        tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        for req in request_iterator:
            if req.meta.filename != "":
                filepath = os.path.join(tmp_dir, req.meta.filename)
                version = req.meta.version
                request_id = req.meta.requestId
                logger.debug(
                    """ReplicateModel | {"reqId": "%s", "version": %d, "filename": "%s"}""",
                    req.meta.requestId, req.meta.version, req.meta.filename)
                continue
            data.extend(req.chunkData)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(data)

        self.actor.set_model_replicate(version, filepath)
        if self.actor.model_loaded.wait(5):
            done = True
            err = ""
        else:
            done = False
            err = "model replicating timeout"
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass

        resp = actor_pb2.ReplicateModelResp(done=done, err=err, requestId=request_id)
        return resp

    def Terminate(self, request, context):
        logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.actor.terminate()
        self.stop_event.set()
        return actor_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        remote_buffer_address: str,
        local_buffer_size: int,
        env: EnvWrapper,
        debug: bool = False
) -> tp.Tuple[grpc.Server, threading.Event]:
    stop_event = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="distActor"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 100),
        ]
    )

    actor_p = ActorProcess(
        local_buffer_size=local_buffer_size,
        env=env,
        remote_buffer_address=remote_buffer_address,
        debug=debug,
    )
    service = ActorService(
        actor=actor_p,
        stop_event=stop_event,
        debug=debug
    )

    actor_pb2_grpc.add_ActorServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info("actor has started at http://localhost:%d", port)
    return server, stop_event


def start_actor_server(
        port: int,
        remote_buffer_address: str,
        local_buffer_size: int,
        env: EnvWrapper,
        debug: bool = False
):
    server, stop_event = _start_server(port, remote_buffer_address, local_buffer_size, env, debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    logger.info("%s server is down", logger.name)
