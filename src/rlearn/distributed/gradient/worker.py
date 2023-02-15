import logging
import multiprocessing as mp
import random
import string
import threading
import typing as tp
from concurrent import futures

import grpc

from rlearn.distributed import tools, base
from rlearn.distributed.gradient import worker_pb2, worker_pb2_grpc
from rlearn.distributed.logger import get_logger
from rlearn.env.env_wrapper import EnvWrapper

_name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
_logger = get_logger(f"actor-{_name}")


class WorkerProcess(base.MulProcess):
    def __init__(
            self,
            env: EnvWrapper,
            weights_conn: mp.connection.Connection,
            gradients_conn: mp.connection.Connection,
            debug: bool = False,
    ):
        super().__init__(
            env=env,
            weights_conn=weights_conn,
            logger=_logger,
            debug=debug
        )
        self.gradients_conn = gradients_conn
        self.ns.send_gradient = False

    def try_sync(self):
        if self.ns.send_gradient:
            gradients = self.trainer.get_gradients()
            self.gradients_conn.send([self.ns.version, gradients])
            with self.lock:
                self.ns.send_gradient = False
            version, weights = self.weights_conn.recv()
            self.trainer.model.set_flat_weights(weights=weights)
            self.ns.version = version
            self.logger.debug("model parameters replaced, version=%d", version)
            self.model_loaded.set()

    def run(self):
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        episode_generator = tools.get_count_generator(self.max_episode)

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

                _a = self.trainer.predict(s)
                a = self.trainer.map_action(_a)

                s_, r, done = self.env.step(a)
                self.trainer.store_transition(s=s, a=_a, r=r, s_=s_, done=done)
                s = s_

                self.try_sync()

                if done:
                    break

            self.logger.debug("ep=%d, total_step=%d", ep, step)


class WorkerService(worker_pb2_grpc.WorkerServicer):
    def __init__(
            self,
            worker: WorkerProcess,
            weights_conn: mp.connection.Connection,
            gradients_conn: mp.connection.Connection,
            stop_event: threading.Event,
            debug=False):
        self.worker = worker
        self.stop_event = stop_event
        self.weights_conn = weights_conn
        self.gradients_conn = gradients_conn

        _logger.setLevel(logging.DEBUG if debug else logging.ERROR)

    def ServiceReady(self, request, context):
        _logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return worker_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request_iterator, context):
        tools.initialize_model(
            request_iterator=request_iterator,
            logger=_logger,
            process=self.worker,
            resp=worker_pb2.StartResp,
        )

    def ReplicateModel(self, request_iterator, context):
        return tools.replicate_model(
            request_iterator=request_iterator,
            logger=_logger,
            model_loaded_event=self.worker.model_loaded,
            weights_conn=self.weights_conn,
            resp=worker_pb2.ReplicateModelResp,
        )

    def GetGradients(self, request, context):
        self.worker.ns.send_gradient = True
        version, gradients = self.gradients_conn.recv()
        return tools.get_iter_values(
            req=worker_pb2.ReplicateModelReq,
            meta=worker_pb2.ModelMeta,
            values=gradients,
            version=version,
            chunk_size=1024,
            request_id=request.requestId,
        )

    def Terminate(self, request, context):
        _logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.worker.terminate()
        self.stop_event.set()
        return worker_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        env: EnvWrapper,
        debug: bool = False
) -> tp.Tuple[grpc.Server, threading.Event]:
    stop_event = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="distWorker"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 100),
        ]
    )
    recv_weights_conn, send_weights_conn = base.mp.Pipe(duplex=False)
    recv_gradients_conn, send_gradients_conn = base.mp.Pipe(duplex=False)
    worker_p = WorkerProcess(
        env=env,
        weights_conn=recv_weights_conn,
        gradients_conn=send_gradients_conn,
        debug=debug,
    )
    service = WorkerService(
        worker=worker_p,
        weights_conn=send_weights_conn,
        gradients_conn=recv_gradients_conn,
        stop_event=stop_event,
        debug=debug
    )

    worker_pb2_grpc.add_WorkerServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    _logger.info("actor has started at http://localhost:%d", port)
    return server, stop_event


def start_actor_server(
        port: int,
        env: EnvWrapper,
        debug: bool = False
):
    server, stop_event = _start_server(port, env, debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    _logger.info("%s server is down", _logger.name)
