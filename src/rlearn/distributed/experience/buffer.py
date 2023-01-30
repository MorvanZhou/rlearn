import inspect
import logging
import threading
import time
import typing as tp
from concurrent import futures

import grpc

from rlearn import replaybuf
from rlearn.distributed import tools
from rlearn.distributed.experience import buffer_pb2, buffer_pb2_grpc
from rlearn.distributed.logger import get_logger
from rlearn.replaybuf.base import BaseReplayBuffer


class ReplayBufferService(buffer_pb2_grpc.ReplayBufferServicer):
    def __init__(
            self,
            stop_event: threading.Event,
            max_size: int,
            buf: tp.Union[str, tp.Type[BaseReplayBuffer]] = "RandomReplayBuffer",
            debug=False
    ):
        self.stop_event = stop_event
        if inspect.isclass(buf):
            name = buf.name
        else:
            name = buf

        self.logger = get_logger("buf")
        self.logger.setLevel(logging.DEBUG if debug else logging.ERROR)

        self.version = 0
        self.is_on_policy = False
        self.replay_buffer = replaybuf.get_buffer_by_name(name=name, max_size=max_size)
        self.is_uploading = False
        self.is_downloading = False

    def ServiceReady(self, request, context):
        self.logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return buffer_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def LearnerSetModelType(self, request, context):
        self.logger.debug(
            """LearnerSetModelType | {"reqId": "%s", "isOnPolicy": "%s"}""", request.requestId, request.isOnPolicy)
        self.is_on_policy = request.isOnPolicy
        return buffer_pb2.LearnerSetVersionResp(done=True, err="", requestId=request.requestId)

    def LearnerSetVersion(self, request, context):
        self.logger.debug(
            """LearnerSetVersion | {"reqId": "%s", "version": %d}""", request.requestId, request.version)
        self.version = request.version
        return buffer_pb2.LearnerSetVersionResp(done=True, err="", requestId=request.requestId)

    def UploadData(self, request, context):
        self.logger.debug("""UploadData | {"reqId": "%s", "version": %d}""", request.requestId, request.version)

        # reject invalided version
        if self.is_on_policy and request.version < self.version:
            self.logger.debug(
                "model isOnPolicy, request version=%d is not equal to last version=%d",
                request.version, self.version)
            return buffer_pb2.UploadDataResp(
                done=True,
                err=f"version='{request.version}' is not aline with learner's '{self.version}'",
                requestId=request.requestId,
            )
        while True:
            if not self.is_downloading:
                break
            self.logger.debug("waiting data downloading and clear")
            time.sleep(0.05)

        batch_size, batch = tools.unpack_transitions(request)
        if batch_size == 0:
            return buffer_pb2.UploadDataResp(done=False, err="no data uploaded", requestId=request.requestId)
        try:
            self.is_uploading = True
            self.replay_buffer.put_batch(**batch)
        except (ValueError, TypeError) as e:
            self.logger.error("UpdateData err: %s", str(e).replace("\n", "\\n"))
            resp = buffer_pb2.UploadDataResp(done=False, err=str(e), requestId=request.requestId)
        else:
            resp = buffer_pb2.UploadDataResp(done=True, err="", requestId=request.requestId)
        finally:
            self.is_uploading = False
        return resp

    def DownloadData(self, request, context):
        self.logger.debug("""DownloadData | {"reqId": "%s", "maxSize": %d}""", request.requestId, request.maxSize)
        while True:
            if not self.is_uploading:
                break
            self.logger.debug("waiting data uploading")
            time.sleep(0.05)

        if self.replay_buffer.is_empty():
            return buffer_pb2.DownloadDataResp(err="no data", requestId=request.requestId)

        max_size = request.maxSize
        if max_size <= 0:
            max_size = None
        self.is_downloading = True
        resp = buffer_pb2.DownloadDataResp(err="", requestId=request.requestId)
        tools.pack_transitions(buffer=self.replay_buffer, interface=resp, max_size=max_size)
        self.replay_buffer.clear()
        self.is_downloading = False
        return resp

    def Stop(self, request, context):
        self.logger.debug("""Stop | {"reqId": "%s"}""", request.requestId)
        self.stop_event.set()
        return buffer_pb2.StopResp(done=True, requestId=request.requestId)


def _start_server(
        port: int,
        max_size: int,
        buf: tp.Union[str, tp.Type[BaseReplayBuffer]] = "RandomReplayBuffer",
        debug: bool = False,
) -> tp.Tuple[grpc.Server, threading.Event]:
    stop_event = threading.Event()
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="distBuffer"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 20),
        ]
    )
    service = ReplayBufferService(stop_event, max_size, buf, debug)
    buffer_pb2_grpc.add_ReplayBufferServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    service.logger.info("replay buffer has started at http://localhost:%d", port)
    return server, stop_event


def start_replay_buffer_server(
        port: int,
        max_size: int,
        buf: tp.Union[str, tp.Type[BaseReplayBuffer]] = "RandomReplayBuffer",
        debug: bool = False,
) -> grpc.Server:
    server, stop_event = _start_server(port, max_size, buf, debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    return server
