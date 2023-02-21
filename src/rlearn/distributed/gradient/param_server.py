import datetime
import json
import logging
import os
import tempfile
import threading
import time
import typing as tp
import zlib
from concurrent import futures

import grpc
import numpy as np

from rlearn.distributed.gradient import param_pb2, param_pb2_grpc
from rlearn.distributed.logger import get_logger
from rlearn.trainer.base import BaseTrainer

logger = get_logger(f"paramServer")


class ParamService(param_pb2_grpc.ParamsServicer):
    def __init__(
            self,
            trainer: BaseTrainer,
            sync_step: int,
            worker_buffer_size: int,
            stop_event: threading.Event,
            worker_buffer_type: str = "RandomReplayBuffer",
            max_ep_step: int = -1,
            max_train_time: int = -1,
            save_dir: str = "",
            save_frequency: int = 0,  # seconds
            debug: bool = False
    ):
        logger.setLevel(logging.DEBUG if debug else logging.ERROR)
        self.trainer = trainer
        self.sync_step = sync_step
        self.worker_buffer_size = worker_buffer_size
        self.worker_buffer_type = worker_buffer_type
        self.max_ep_step = max_ep_step
        self.stop_event = stop_event
        self.max_train_time = max_train_time
        self.stop = False
        self.workers_stop = {}
        self.start_time = time.time()

        if save_dir is None or save_dir == "":
            save_dir = os.path.join("savedModel", trainer.name)
        self.save_dir = os.path.normpath(save_dir)
        self.save_frequency = int(save_frequency)
        self._last_save_time = time.time()

    def ServiceReady(self, request, context):
        logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return param_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def iter_start(self, request_id, chunk_size: int = 1024):
        if self.trainer.model.action_transformer is None:
            at = []
        else:
            at = self.trainer.model.action_transformer.params
        yield param_pb2.StartResp(
            meta=param_pb2.StartMeta(
                syncStep=self.sync_step,
                trainerType=self.trainer.name,
                bufferType=self.worker_buffer_type,
                bufferSize=self.worker_buffer_size,
                maxEpisodeStep=self.max_ep_step,
                actionTransform=json.dumps(at),
                requestId=request_id,
                batchSize=self.trainer.batch_size,
                gamma=self.trainer.gamma
            )
        )
        path = os.path.join(tempfile.gettempdir(), f"param_server_pb.zip")
        self.trainer.save_model(path)
        with open(path, mode="rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if chunk:
                    yield param_pb2.StartResp(chunkData=chunk)
                else:  # The chunk was empty, which means we're at the end of the file
                    return

    def Start(self, request, context):
        logger.debug("""Start | {"reqId": "%s"}""", request.requestId)
        self.workers_stop[context.peer()] = False
        return self.iter_start(request_id=request.requestId, chunk_size=1024)

    def iter_weights(self, request_id: str, chunk_size: int = 1024):
        yield param_pb2.SyncResp(
            meta=param_pb2.WeightsMeta(
                requestId=request_id,
                stop=self.stop,
            )
        )
        weights = self.trainer.model.get_flat_weights()
        b_weights = weights.astype(np.float32).tobytes()
        compressed_data = zlib.compress(b_weights)
        for i in range(0, len(compressed_data), chunk_size):
            yield param_pb2.SyncResp(chunkData=compressed_data[i: i + chunk_size])

        if all(self.workers_stop.values()):
            self.stop_event.set()

    def Sync(self, request_iterator, context):
        if self.max_train_time > 0 and time.time() - self.start_time > self.max_train_time:
            self.stop = True
            self.workers_stop[context.peer()] = True

        data = bytearray()
        request_id = ""
        for r in request_iterator:
            if r.chunkData == b"":
                request_id = r.requestId
                logger.debug("""Sync | {"reqId": "%s"}""", request_id)
                continue
            data.extend(r.chunkData)
        d_data = zlib.decompress(data)
        gradients = np.frombuffer(d_data, dtype=np.float32)
        self.trainer.apply_flat_gradients(gradients=gradients)
        self.try_save()
        return self.iter_weights(request_id=request_id, chunk_size=1024)

    def try_save(self):
        if self.save_frequency <= 0:
            return
        if self.save_frequency > time.time() - self._last_save_time:
            self._last_save_time = time.time()
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.save_dir, current_time)
            self.trainer.save_model(path)
            logger.debug("save model to %s", path)

    def Terminate(self, request, context):
        logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.stop = True
        self.stop_event.set()
        return param_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        trainer: BaseTrainer,
        sync_step: int,
        worker_buffer_size: int,
        worker_buffer_type: str = "RandomReplayBuffer",
        max_train_time: int = -1,
        max_ep_step: int = -1,
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
    service = ParamService(
        trainer=trainer,
        sync_step=sync_step,
        worker_buffer_size=worker_buffer_size,
        worker_buffer_type=worker_buffer_type,
        max_ep_step=max_ep_step,
        stop_event=stop_event,
        max_train_time=max_train_time,
        debug=debug
    )

    param_pb2_grpc.add_ParamsServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info("param server has started at http://localhost:%d", port)
    return server, stop_event


def start_param_server(
        port: int,
        trainer: BaseTrainer,
        sync_step: int,
        worker_buffer_size: int,
        worker_buffer_type: str = "RandomReplayBuffer",
        max_train_time: int = -1,
        max_ep_step: int = -1,
        debug: bool = False
):
    server, stop_event = _start_server(
        port=port,
        trainer=trainer,
        sync_step=sync_step,
        worker_buffer_size=worker_buffer_size,
        worker_buffer_type=worker_buffer_type,
        max_train_time=max_train_time,
        max_ep_step=max_ep_step,
        debug=debug,
    )
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    logger.info("%s server is down", logger.name)
