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

logger = get_logger("buf")


class ReplayBufferService(buffer_pb2_grpc.ReplayBufferServicer):

    def __init__(
            self,
            stop_event: threading.Event,
            debug=False
    ):
        self.stop_event = stop_event
        logger.setLevel(logging.DEBUG if debug else logging.ERROR)

        self.version = 0
        self.is_on_policy = False
        self.is_uploading = False
        self.replay_buffer: tp.Optional[BaseReplayBuffer] = None

    def ServiceReady(self, request, context):
        logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return buffer_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def InitBuf(self, request, context):
        logger.debug(
            """InitBuf | {"reqId": "%s", "isOnPolicy": %s, "bufferType": "%s", "bufferSize": %d}""",
            request.requestId, request.isOnPolicy, request.bufferType, request.bufferSize)
        self.is_on_policy = request.isOnPolicy
        self.replay_buffer = replaybuf.get_buffer_by_name(name=request.bufferType, max_size=request.bufferSize)
        return buffer_pb2.InitBufResp(done=True, err="", requestId=request.requestId)

    def LearnerSetVersion(self, request, context):
        logger.debug(
            """LearnerSetVersion | {"reqId": "%s", "version": %d}""", request.requestId, request.version)
        self.version = request.version
        return buffer_pb2.LearnerSetVersionResp(done=True, err="", requestId=request.requestId)

    def UploadData(self, request_iterator, context):
        batch_size, batch, version, request_id = tools.unpack_uploaded_transitions(request_iterator=request_iterator)

        # reject invalided version
        if self.is_on_policy and version < self.version:
            logger.debug(
                "model isOnPolicy, request version=%d is not equal to last version=%d",
                version, self.version)
            return buffer_pb2.UploadDataResp(
                done=True,
                err=f"version='{version}' is not aline with learner's '{self.version}'",
                requestId=request_id,
            )

        if batch_size == 0:
            return buffer_pb2.UploadDataResp(done=False, err="no data uploaded", requestId=request_id)
        try:
            self.is_uploading = True
            self.replay_buffer.put_batch(**batch)
        except (ValueError, TypeError) as e:
            logger.error("UpdateData err: %s", str(e).replace("\n", "\\n"))
            resp = buffer_pb2.UploadDataResp(done=False, err=str(e), requestId=request_id)
        else:
            resp = buffer_pb2.UploadDataResp(done=True, err="", requestId=request_id)
        finally:
            self.is_uploading = False
        return resp

    def DownloadData(self, request, context):
        logger.debug("""DownloadData | {"reqId": "%s", "maxSize": %d}""", request.requestId, request.maxSize)
        while True:
            if not self.is_uploading:
                break
            logger.debug("waiting data uploading")
            time.sleep(0.05)

        max_size = request.maxSize
        if max_size <= 0:
            max_size = None
        resp_iter = tools.pack_transitions_for_downloading(
            buffer=self.replay_buffer,
            max_size=max_size,
            request_id=request.requestId,
            chunk_size=1024,
        )
        return resp_iter

    def Stop(self, request, context):
        logger.debug("""Stop | {"reqId": "%s"}""", request.requestId)
        self.stop_event.set()
        return buffer_pb2.StopResp(done=True, requestId=request.requestId)


def _start_server(
        port: int,
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
    service = ReplayBufferService(stop_event=stop_event, debug=debug)
    buffer_pb2_grpc.add_ReplayBufferServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info("replay buffer has started at http://localhost:%d", port)
    return server, stop_event


def start_replay_buffer_server(
        port: int,
        debug: bool = False,
):
    server, stop_event = _start_server(port=port, debug=debug)
    stop_event.wait()
    server.stop(None)
    server.wait_for_termination()
    logger.info("reply buffer server is down")
