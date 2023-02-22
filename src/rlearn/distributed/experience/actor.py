import logging
import multiprocessing as mp
import os
import random
import string
import threading
import typing as tp
from concurrent import futures
from multiprocessing.connection import Connection

import grpc
from rlearn.distributed import tools
from rlearn.distributed.experience import buffer_pb2_grpc, actor_pb2_grpc, actor_pb2
from rlearn.distributed.logger import get_logger
from rlearn.env.env_wrapper import EnvWrapper
from rlearn.trainer.base import BaseTrainer
from rlearn.trainer.tools import get_trainer_by_name, set_trainer_action_transformer

# linux default is fork, force set to spawn
mp = mp.get_context('spawn')


class ActorProcess(mp.Process):
    def __init__(
            self,
            weights_conn: Connection,
            env: EnvWrapper,
            logger: logging.Logger = None,
            remote_buffer_address: tp.Optional[str] = None,
            debug: bool = False,
    ):
        super().__init__()
        if logger is None:
            logger = get_logger("actor")
        self.logger = logger
        self.weights_conn = weights_conn
        self.debug = debug
        self.trainer_type = ""
        self.buffer_type = ""
        self.buffer_size = 0
        self.training_request_id = ""
        self.init_model_pb_path = ""
        self.trainer: tp.Optional[BaseTrainer] = None
        self.env: EnvWrapper = env
        self.max_episode = 0
        self.max_episode_step = 0
        self.action_transform = None

        mgr = mp.Manager()
        self.ns = mgr.Namespace()

        self.ns.version = 0
        self.ns.episode_num = 0
        self.ns.episode_step_num = 0
        self.ns.exit = False
        self.lock = mgr.Lock()
        self.model_loaded = mgr.Event()
        self.model_loaded.clear()

        self.buffer_address = remote_buffer_address

    def init_params(
            self,
            trainer_type: str,
            model_pb_path: str,
            init_version: int,
            request_id: str,
            buffer_size: int,
            buffer_type: str = "RandomReplayBuffer",
            max_episode: int = 0,
            max_episode_step: int = 0,
            action_transform: list = None,
    ):
        self.trainer_type = trainer_type
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.training_request_id = request_id
        self.init_model_pb_path = model_pb_path
        if action_transform is not None:
            self.action_transform = action_transform
        self.ns.version = init_version
        self.max_episode = 0 if max_episode is None else max_episode
        self.max_episode_step = 0 if max_episode_step is None else max_episode_step

    def try_replicate_model(self):
        if not self.weights_conn.closed and self.weights_conn.poll():
            try:
                version, weights = self.weights_conn.recv()
            except EOFError as err:
                self.logger.error(str(err))
                return
            self.trainer.model.set_flat_weights(weights=weights)
            with self.lock:
                self.ns.version = version
            self.logger.debug("model parameters replaced, version=%d", self.ns.version)
            self.model_loaded.set()

    def send_data_to_remote_buffer(self, buf_stub):
        req_iter = tools.pack_transitions_for_uploading(
            buffer=self.trainer.replay_buffer,
            version=self.ns.version,
            request_id=self.training_request_id,
        )
        resp = buf_stub.UploadData(req_iter)
        if not resp.done:
            raise ValueError(f"grpc upload data to buffer err: {resp.err}")

    def set_trainer(self):
        self.trainer = get_trainer_by_name(self.trainer_type)
        self.trainer.set_params(min_epsilon=0.1, epsilon_decay=1e-3)
        self.trainer.set_replay_buffer(max_size=self.buffer_size, buf=self.buffer_type)
        if self.action_transform is not None:
            set_trainer_action_transformer(self.trainer, self.action_transform)

        self.trainer.load_model(self.init_model_pb_path)
        self.logger.debug("initial model is set from path='%s'", self.init_model_pb_path)
        os.remove(self.init_model_pb_path)
        self.model_loaded.set()

    def run(self):
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        episode_generator = tools.get_count_generator(self.max_episode)

        if self.buffer_address is None or self.buffer_address.strip() == "":
            buf_stub = None
            channel = None
        else:
            channel = grpc.insecure_channel(self.buffer_address)
            buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        self.set_trainer()

        with self.lock:
            self.ns.episode_num = 0

        for ep in episode_generator:
            with self.lock:
                self.ns.episode_num = ep
            s = self.env.reset()

            step = 0
            ep_r = 0
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
                ep_r += r
                if self.trainer.replay_buffer.is_full() and buf_stub is not None:
                    self.send_data_to_remote_buffer(buf_stub)
                    self.trainer.replay_buffer.clear()
                if done or self.ns.exit:
                    break

            if self.ns.exit:
                break

            self.logger.info("ep=%d, total_step=%d, ep_r=%.3f", ep, step, ep_r)

        if channel is not None:
            channel.close()

        self.weights_conn.close()


class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(
            self,
            actor: ActorProcess,
            weights_conn: Connection,
            stop_event: threading.Event,
            logger: logging.Logger,
            debug: bool = False,
    ):
        self.actor = actor
        self.weights_conn = weights_conn
        self.stop_event = stop_event
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.ERROR)
        with grpc.insecure_channel(self.actor.buffer_address) as channel:
            timeout = 15
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                raise ValueError(f"connect replay buffer at {self.actor.buffer_address} timeout: {timeout}")
        self.logger.debug("connected to buffer %s", self.actor.buffer_address)

    def ServiceReady(self, request, context):
        self.logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return actor_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request_iterator, context):
        return tools.initialize_model(
            request_iterator=request_iterator,
            logger=self.logger,
            process=self.actor,
            resp=actor_pb2.StartResp,
        )

    def ReplicateModel(self, request_iterator, context):
        return tools.replicate_model(
            request_iterator=request_iterator,
            logger=self.logger,
            model_loaded_event=self.actor.model_loaded,
            weights_conn=self.weights_conn,
        )

    def Terminate(self, request, context):
        self.logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.actor.ns.exit = True
        self.actor.join()
        self.stop_event.set()
        self.weights_conn.close()
        return actor_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        remote_buffer_address: str,
        env: EnvWrapper,
        name: str = "",
        debug: bool = False
) -> tp.Tuple[grpc.Server, threading.Event, logging.Logger]:
    stop_event = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="distActor"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 100),
        ]
    )
    if name == "":
        name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
    logger = get_logger(f"actor-{name}")
    recv_conn, send_conn = mp.Pipe(duplex=False)
    actor_p = ActorProcess(
        weights_conn=recv_conn,
        env=env,
        logger=logger,
        remote_buffer_address=remote_buffer_address,
        debug=debug,
    )
    service = ActorService(
        actor=actor_p,
        weights_conn=send_conn,
        stop_event=stop_event,
        logger=logger,
        debug=debug
    )

    actor_pb2_grpc.add_ActorServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info("actor has started at http://localhost:%d", port)
    return server, stop_event, logger


def start_actor_server(
        port: int,
        remote_buffer_address: str,
        env: EnvWrapper,
        name: str = "",
        debug: bool = False
):
    server, stop_event, logger = _start_server(port, remote_buffer_address, env, name=name, debug=debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    logger.info("%s server is down", logger.name)
