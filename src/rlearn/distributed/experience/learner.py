import datetime
import logging
import os
import tempfile
import threading
import time
import typing as tp
from uuid import uuid4

import grpc

from rlearn import replaybuf
from rlearn.distributed import tools, logger
from rlearn.distributed.experience import buffer_pb2_grpc, buffer_pb2, actor_pb2_grpc, actor_pb2
from rlearn.trainer.base import BaseTrainer


class Learner:
    def __init__(
            self,
            trainer: BaseTrainer,
            remote_buffer_address: str,
            remote_buffer_size: int,
            actor_buffer_size: int,
            actors_address: tp.Sequence[str],
            remote_buffer_type: str = "RandomReplayBuffer",
            save_dir: str = "",
            save_frequency: int = 0,  # seconds
            debug: bool = False,
    ):
        self.trainer: BaseTrainer = trainer
        self.debug: bool = debug
        self.logger = logger.get_logger("leaner")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        self.actor_buffer_size: int = actor_buffer_size
        self.actor_buffer_type: str = "RandomReplayBuffer"

        self.actors_channel: tp.Dict[str, grpc.Channel] = {
            address: grpc.insecure_channel(address) for address in actors_address
        }
        self.actors_stub: tp.Dict[str, actor_pb2_grpc.ActorStub] = {
            address: actor_pb2_grpc.ActorStub(channel=channel)
            for address, channel in self.actors_channel.items()
        }
        self.version_count = 0

        self.buffer_address: str = remote_buffer_address
        self.buffer_size = remote_buffer_size
        if remote_buffer_type not in replaybuf.tools.get_all_buffers():
            raise KeyError(f"replay buffer name '{remote_buffer_type}' is not found,"
                           f" please use one of these {list(replaybuf.tools.get_all_buffers().keys())}")
        self.buffer_type = remote_buffer_type
        self.buffer_channel = grpc.insecure_channel(self.buffer_address)
        self.buffer_stub = buffer_pb2_grpc.ReplayBufferStub(
            channel=self.buffer_channel
        )
        self.keep_download_data = True

        if save_dir is None or save_dir == "":
            save_dir = os.path.join("savedModel", trainer.name)
        self.save_dir = os.path.normpath(save_dir)
        self.save_frequency = int(save_frequency)
        self._last_save_time = time.time()

    def check_actors_ready(self):
        timeout = 15
        for addr in self.actors_stub.keys():
            try:
                grpc.channel_ready_future(self.actors_channel[addr]).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                raise ValueError(f"connect actor at {addr} timeout: {timeout}")

        self.logger.debug("actors server ready")

    def check_buffer_ready(self):
        timeout = 15
        try:
            grpc.channel_ready_future(self.buffer_channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            raise ValueError(f"connect replay buffer at {self.buffer_address} timeout: {timeout}")
        self.logger.debug("connected to buffer %s", self.buffer_address)

    def init_actors(self):
        path = os.path.join(tempfile.gettempdir(), f"params_v{self.version_count}.zip")
        self.trainer.save_model(path)
        futures = {}
        if self.trainer.model.action_transformer is None:
            at = []
        else:
            at = self.trainer.model.action_transformer.params

        for addr, stub in self.actors_stub.items():
            req_id = str(uuid4())
            try:
                resp_future = stub.Start.future(
                    tools.read_pb_iterfile(
                        filepath=path,
                        trainer_type=self.trainer.name,
                        buffer_type=self.actor_buffer_type,
                        buffer_size=self.actor_buffer_size,
                        max_episode=0,
                        max_episode_step=0,
                        action_transform=at,
                        version=self.version_count,
                        request_id=req_id,
                    )
                )
                futures[addr] = resp_future
            except grpc.RpcError as e:
                self.logger.error("actor start err: %e, addr='%s', requestId='%s'", str(e), addr, req_id)

        for addr, resp_future in futures.items():
            resp = resp_future.result()
            if not resp.done:
                raise ValueError(f"actor start err: {resp.err}, addr='{addr}', requestId='{resp.requestId}'")

        self.logger.debug("init actors with version=%d", self.version_count)

    def replicate_actors_model(self):
        t0 = time.perf_counter()
        weights = self.trainer.model.get_flat_weights()
        req_id = str(uuid4())
        futures = {}
        for addr, stub in self.actors_stub.items():
            try:
                resp_future = stub.ReplicateModel.future(
                    tools.get_iter_values(
                        msg_handler=actor_pb2.ReplicateModelReq,
                        values=weights,
                        version=self.version_count,
                        request_id=req_id
                    ))
                futures[addr] = resp_future
            except grpc.RpcError as e:
                self.logger.error(
                    "worker replicate params err: %e, addr='%s', "
                    "version=%d, requestId='%s'", str(e), addr, self.version_count, req_id)

        for addr, resp_future in futures.items():
            resp = resp_future.result()
            if not resp.done:
                raise ValueError(
                    f"actor replicate params err: {resp.err}, "
                    f"addr='{addr}', version={self.version_count}, requestId='{req_id}'")
        t1 = time.perf_counter()
        self.logger.debug(
            "replicate actors params, version=%d, spend=%.3fs, reqId='%s'",
            self.version_count, t1 - t0, req_id)

    def stop_actors(self):
        # stop actors
        req_id = str(uuid4())
        for addr, stub in self.actors_stub.items():
            resp = stub.Terminate(actor_pb2.TerminateReq(requestId=req_id))
            if not resp.done:
                self.logger.error(
                    "actor not terminated, err: %s, addr='%s', reqId='%s'", resp.err, addr, resp.requestId)
        for c in self.actors_channel.values():
            c.close()

    def init_buffer(self):
        req_id = str(uuid4())
        resp = self.buffer_stub.InitBuf(
            buffer_pb2.InitBufReq(
                isOnPolicy=self.trainer.is_on_policy,
                bufferSize=self.buffer_size,
                bufferType=self.buffer_type,
                requestId=req_id
            ))
        if not resp.done:
            raise ValueError(f"buffer set modelType err: {resp.err}, requestId='{resp.requestId}'")

        resp = self.buffer_stub.LearnerSetVersion(
            buffer_pb2.LearnerSetVersionReq(
                version=self.version_count,
                requestId=req_id
            ))
        if not resp.done:
            raise ValueError(f"buffer set init version err: {resp.err}, requestId='{resp.requestId}'")
        self.logger.debug(
            "init buffer with version=%d, modelTypeOnPolicy=%s, reqId='%s'",
            self.version_count, self.trainer.is_on_policy, req_id)

    def try_save(self):
        if self.save_frequency <= 0:
            return
        if self.save_frequency < time.time() - self._last_save_time:
            self._last_save_time = time.time()
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.save_dir, current_time)
            self.trainer.save_model(path)
            self.logger.debug("save model to %s", path)

    def download_data(self, lock, max_size=1000):
        while self.keep_download_data:
            time.sleep(1)
            req_id = str(uuid4())
            t0 = time.perf_counter()
            resp_iter = self.buffer_stub.DownloadData(buffer_pb2.DownloadDataReq(
                maxSize=max_size,
                requestId=req_id,
            ))
            batch_size, batch, err, request_id = tools.unpack_downloaded_transitions(resp_iter=resp_iter)
            if err != "":
                self.logger.debug(
                    'UploadData | {"reqId": "%s", "error": "%s"}',
                    request_id,
                    err,
                )
                continue
            t1 = time.perf_counter()
            if batch_size == 0:
                self.logger.debug("download empty data, retry")
                continue
            with lock:
                self.trainer.replay_buffer.put_batch(**batch)
            t2 = time.perf_counter()
            self.logger.debug("downloaded size=%d, request=%.3fs, unpack=%.3fs, reqId='%s'",
                              batch_size, t1 - t0, t2 - t1, req_id)
        resp = self.buffer_stub.Stop(buffer_pb2.StopReq(requestId=str(uuid4())))
        if not resp.done:
            raise ValueError("buffer not exits")
        self.buffer_channel.close()

    def set_buffer_version(self):
        req_id = str(uuid4())
        resp = self.buffer_stub.LearnerSetVersion(
            buffer_pb2.LearnerSetVersionReq(
                version=self.version_count,
                requestId=req_id
            ))
        if not resp.done:
            raise ValueError(
                f"buffer set version err: {resp.err}, version={self.version_count}, requestId='{req_id}'")

    def run(
            self,
            max_train_time: int = -1,
            max_ep_step: int = -1,
            download_max_size: int = 1000,
            replicate_step: tp.Optional[int] = None,
    ):
        self.check_buffer_ready()
        self.check_actors_ready()
        self.init_buffer()
        self.init_actors()

        lock = threading.Lock()
        td = threading.Thread(target=self.download_data, kwargs=dict(lock=lock, max_size=download_max_size))
        td.start()

        if replicate_step is None and not self.trainer.is_on_policy:
            replicate_step = 100
        count = 0
        start_time = time.time()
        ep = 0
        stop = False
        while not stop:
            while self.trainer.replay_buffer.current_loading_point < 100:
                time.sleep(0.5)
                continue
            if max_ep_step <= 0:
                if self.trainer.replay_buffer.is_full():
                    _ep_step = self.trainer.replay_buffer.max_size // self.trainer.batch_size
                else:
                    _ep_step = self.trainer.replay_buffer.current_loading_point // self.trainer.batch_size
            else:
                _ep_step = max_ep_step
            t0 = time.perf_counter()
            for _step in range(_ep_step):
                res = self.trainer.train_batch()
                if self.trainer.is_on_policy:
                    if res.model_replaced:
                        self.version_count += 1
                        self.set_buffer_version()
                        self.replicate_actors_model()
                else:
                    if count % replicate_step == 0:
                        self.version_count += 1
                        self.set_buffer_version()
                        self.replicate_actors_model()
                    count += 1

                self.try_save()
                if max_train_time > 0 and time.time() - start_time > max_train_time:
                    stop = True
                    break
            t1 = time.perf_counter()
            self.logger.debug("trained %d times in ep=%d, spend=%.2fs", _ep_step, ep, t1 - t0)
            ep += 1

        # stop actors
        self.stop_actors()

        # stop downloading data
        self.keep_download_data = False
        td.join()
