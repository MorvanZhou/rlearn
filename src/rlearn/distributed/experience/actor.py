import logging
import multiprocessing as mp
import random
import string
import threading
import typing as tp
from concurrent import futures

import grpc

from rlearn.distributed import tools, base
from rlearn.distributed.experience import actor_pb2, actor_pb2_grpc, buffer_pb2_grpc, buffer_pb2
from rlearn.distributed.logger import get_logger
from rlearn.env.env_wrapper import EnvWrapper

_name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
_logger = get_logger(f"actor-{_name}")


class ActorProcess(base.MulProcess):
    def __init__(
            self,
            weights_conn: mp.connection.Connection,
            env: EnvWrapper,
            remote_buffer_address: tp.Optional[str] = None,
            debug: bool = False,
    ):
        super().__init__(
            env=env,
            weights_conn=weights_conn,
            logger=_logger,
            debug=debug
        )

        self.buffer_address = remote_buffer_address

    def try_replicate_model(self):
        if not self.weights_conn.closed and self.weights_conn.poll():
            version, shapes, weights = self.weights_conn.recv()
            self.trainer.model.set_shapes_weights(shapes=shapes, weights=weights)
            with self.lock:
                self.ns.version = version
            self.logger.debug("model parameters replaced, version=%d", self.ns.version)
            self.model_loaded.set()

    def send_data_to_remote_buffer(self, buf_stub):
        req = buffer_pb2.UploadDataReq(version=self.ns.version, requestId=self.training_request_id)
        tools.pack_transitions(self.trainer.replay_buffer, req)
        resp = buf_stub.UploadData(req)
        if not resp.done:
            raise ValueError(f"grpc upload data to buffer err: {resp.err}")

    def run(self):
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

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
                if done or self.ns.exit:
                    break

            if self.ns.exit:
                break

            self.logger.debug("ep=%d, total_step=%d", ep, step)

        if channel is not None:
            channel.close()

        self.weights_conn.close()


class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(
            self,
            actor: ActorProcess,
            weights_conn: mp.connection.Connection,
            stop_event: threading.Event,
            debug: bool = False,
    ):
        self.actor = actor
        self.weights_conn = weights_conn
        self.stop_event = stop_event

        _logger.setLevel(logging.DEBUG if debug else logging.ERROR)
        with grpc.insecure_channel(self.actor.buffer_address) as channel:
            timeout = 15
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                raise ValueError(f"connect replay buffer at {self.actor.buffer_address} timeout: {timeout}")
        _logger.debug("connected to buffer %s", self.actor.buffer_address)

    def ServiceReady(self, request, context):
        _logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return actor_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request_iterator, context):
        return tools.initialize_model(
            request_iterator=request_iterator,
            logger=_logger,
            process=self.actor,
            resp=actor_pb2.StartResp,
        )

    def ReplicateModel(self, request_iterator, context):
        return tools.replicate_model(
            request_iterator=request_iterator,
            logger=_logger,
            model_loaded_event=self.actor.model_loaded,
            weights_conn=self.weights_conn,
            resp=actor_pb2.ReplicateModelResp,
        )

    def Terminate(self, request, context):
        _logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.weights_conn.close()
        self.actor.ns.exit = True
        self.actor.join()
        self.stop_event.set()
        return actor_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        remote_buffer_address: str,
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
    recv_conn, send_conn = mp.Pipe(duplex=False)
    actor_p = ActorProcess(
        weights_conn=recv_conn,
        env=env,
        remote_buffer_address=remote_buffer_address,
        debug=debug,
    )
    service = ActorService(
        actor=actor_p,
        weights_conn=send_conn,
        stop_event=stop_event,
        debug=debug
    )

    actor_pb2_grpc.add_ActorServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    _logger.info("actor has started at http://localhost:%d", port)
    return server, stop_event


def start_actor_server(
        port: int,
        remote_buffer_address: str,
        env: EnvWrapper,
        debug: bool = False
):
    server, stop_event = _start_server(port, remote_buffer_address, env, debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    _logger.info("%s server is down", _logger.name)
