import logging
import time
import inspect
import typing as tp
from concurrent import futures

import grpc

from rlearn import replaybuf
from rlearn.replaybuf.base import BaseReplayBuffer
from rlearn.distribute import buffer_pb2, buffer_pb2_grpc, tools
from rlearn.distribute.logger import get_logger


class ReplayBufferService(buffer_pb2_grpc.ReplayBufferServicer):
    def __init__(
            self,
            max_size: int,
            buf: tp.Union[str, tp.Type[BaseReplayBuffer]] = "RandomReplayBuffer",
            debug=False
    ):
        if inspect.isclass(buf):
            name = buf.name
        else:
            name = buf

        self.logger = get_logger("buf")
        self.logger.setLevel(logging.DEBUG if debug else logging.ERROR)

        self.replay_buffer = replaybuf.get_buffer_by_name(name=name, max_size=max_size)
        self.is_uploading = False

    def ServiceReady(self, request, context):
        self.logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return buffer_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def UploadData(self, request, context):
        self.logger.debug("""UploadData | {"reqId": "%s"}""", request.requestId)
        s, a, r, s_ = tools.unpack_transitions(request)
        try:
            self.is_uploading = True
            self.replay_buffer.put_batch(s, a, r, s_)
        except (ValueError, TypeError) as e:
            self.logger.error("UpdateData err: %s", str(e).replace("\n", "\\n"))
            resp = buffer_pb2.UploadDataResp(done=False, err=str(e), requestId=request.requestId)
        else:
            resp = buffer_pb2.UploadDataResp(done=True, err="", requestId=request.requestId)
        finally:
            self.is_uploading = False
        return resp

    def DownloadData(self, request, context):
        self.logger.debug("""DownloadData | {"reqId": "%s"}""", request.requestId)
        while True:
            if not self.is_uploading:
                break
            self.logger.debug("waiting data uploading")
            time.sleep(0.05)
        max_size = request.maxSize
        if max_size <= 0:
            max_size = None
        resp = buffer_pb2.DownloadDataResp(err="", requestId=request.requestId)
        tools.pack_transitions(buffer=self.replay_buffer, interface=resp, max_size=max_size)
        return resp


def start_replay_buffer_server(
        port,
        max_size: int,
        buf: tp.Union[str, tp.Type[BaseReplayBuffer]] = "RandomReplayBuffer",
        debug: bool = False
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="s"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 20),
        ]
    )
    service = ReplayBufferService(max_size, buf, debug)
    buffer_pb2_grpc.add_ReplayBufferServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    service.logger.info("python grpc has started at http://127.0.0.1:%s", port)
    return server
