import logging
import time
import typing
from concurrent import futures

import grpc

from tipealgs.ml.rl import replaybuf
from tipealgs.ml.rl.distributed import buffer_pb2, buffer_pb2_grpc
from tipecommon import matrix
from tipecommon.tools import get_default_format_logger


class BufferService(buffer_pb2_grpc.ReplayBufferServicer):
    def __init__(self, buf_name: str, max_size: int, debug=False):
        self.logger = get_default_format_logger("distBuffer")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

        self.replay_buffer = replaybuf.get_buffer_by_name(name=buf_name, max_size=max_size)
        self.is_uploading = False

    def ServiceReady(self, _, context):
        self.logger.debug("ServiceReady")
        return buffer_pb2.ServiceReadyResp(ready=True)

    def UploadData(self, request, context):
        self.logger.debug("UploadData")
        s, a, r, s_ = matrix.unpack_transitions(request)
        try:
            self.is_uploading = True
            self.replay_buffer.put_batch(s, a, r, s_)
        except (ValueError, TypeError) as e:
            self.logger.error("UpdateData err: %s", str(e).replace("\n", "\\n"))
            resp = buffer_pb2.UploadDataResp(done=False, err=str(e))
        else:
            resp = buffer_pb2.UploadDataResp(done=True, err="")
        finally:
            self.is_uploading = False
        return resp

    def DownloadData(self, request, context):
        self.logger.debug("DownloadData")
        while True:
            if not self.is_uploading:
                break
            self.logger.debug("waiting data uploading")
            time.sleep(0.05)
        max_size = request.maxSize
        if max_size <= 0:
            max_size = None
        resp = buffer_pb2.DownloadDataResp(err="")
        matrix.pack_transitions(buffer=self.replay_buffer, interface=resp, max_size=max_size)
        return resp


def start_server(port, buf_name: str, max_size: int, debug: bool = False) -> typing.Tuple[grpc.Server, BufferService]:
    _grpc_server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="s"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 20),
        ]
    )
    _thread_service = BufferService(buf_name, max_size, debug)
    buffer_pb2_grpc.add_ReplayBufferServicer_to_server(_thread_service, _grpc_server)
    _grpc_server.add_insecure_port(f'[::]:{port}')
    _grpc_server.start()
    _thread_service.logger.info("python grpc has started at http://127.0.0.1:%s", port)
    return _grpc_server, _thread_service
