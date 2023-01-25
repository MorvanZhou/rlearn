import datetime
import logging
import os
import threading
import time
import typing as tp
from uuid import uuid4

import grpc

from rlearn.distribute import logger, tools
from rlearn.distribute.experience import actor_pb2_grpc, buffer_pb2_grpc, actor_pb2, buffer_pb2
from rlearn.trainer.base import BaseTrainer


class Learner:
    def __init__(
            self,
            trainer: BaseTrainer,
            remote_buffer_address: str,
            remote_actors_address: tp.Sequence[str],
            result_dir: str = None,
            debug: bool = False,
    ):
        self.debug = debug
        self.logger = logger.get_logger("learner")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        self.trainer: BaseTrainer = trainer
        self.buffer_address: str = remote_buffer_address

        self.actors_channel: tp.Dict[str, grpc.Channel] = {
            address: grpc.insecure_channel(address) for address in remote_actors_address
        }
        self.actors_stub: tp.Dict[str, actor_pb2_grpc.ActorStub] = {
            address: actor_pb2_grpc.ActorStub(channel=channel)
            for address, channel in self.actors_channel.items()
        }
        self.buffer_stub = buffer_pb2_grpc.ReplayBufferStub(
            channel=grpc.insecure_channel(self.buffer_address)
        )
        self.version_count = 0
        self.keep_download_data = True

        if result_dir is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
            result_dir = os.path.join("training_results", current_time)
        self.result_dir = os.path.normpath(result_dir)

    def check_actors_buffer_ready(self):
        for _ in range(10):
            try:
                resp = self.buffer_stub.ServiceReady(buffer_pb2.ServiceReadyReq(requestId=str(uuid4())))
                if not resp.ready:
                    raise ValueError(f"replay buffer at {self.buffer_address}"
                                     f" not ready: {resp.err}, requestId='{resp.requestId}'")
            except grpc.RpcError:
                self.logger.debug("waiting for buffer (%s)", self.buffer_address)
                time.sleep(1)
        self.logger.debug("connected to buffer %s", self.buffer_address)

        for addr, stub in self.actors_stub.items():
            for _ in range(10):
                try:
                    resp = stub.ServiceReady(actor_pb2.ServiceReadyReq(requestId=str(uuid4())))
                    if not resp.ready:
                        raise ValueError(f"actor at {addr} not ready: {resp.err}, requestId='{resp.requestId}'")
                    self.logger.debug("connected to actor %s", addr)
                    break
                except grpc.RpcError:
                    self.logger.debug("waiting for actor (%s)", addr)
                    time.sleep(1)

        self.logger.debug("actors server ready")

    def send_init_data(self):
        req_id = str(uuid4())

        resp = self.buffer_stub.LearnerSetModelType(
            buffer_pb2.LearnerSetModelTypeReq(
                isOnPolicy=self.trainer.is_on_policy,
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

        path = os.path.join(self.result_dir, f"params_v{self.version_count}.zip")
        self.trainer.save_model(path)
        futures = {}
        for addr, stub in self.actors_stub.items():
            try:
                resp_future = stub.Start.future(
                    tools.read_pb_iterfile(
                        filepath=path,
                        trainer_type=self.trainer.name,
                        max_episode=0,
                        max_episode_step=0,
                        version=self.version_count,
                        request_id=str(uuid4()),
                    )
                )
                futures[addr] = resp_future
            except grpc.RpcError as e:
                self.logger.error("actor start err: %e, addr='%s', requestId='%s'", str(e), addr, resp.requestId)

        for addr, resp_future in futures.items():
            resp = resp_future.result()
            if not resp.done:
                raise ValueError(f"actor start err: {resp.err}, addr='{addr}', requestId='{resp.requestId}'")

        self.logger.debug("init actors with version=%d, reqId='%s'", self.version_count, req_id)
        self.version_count += 1

    def download_data(self, lock, max_size=1000):
        while self.keep_download_data:
            time.sleep(1)
            req_id = str(uuid4())
            t0 = time.perf_counter()
            resp = self.buffer_stub.DownloadData(buffer_pb2.DownloadDataReq(
                maxSize=max_size,
                requestId=req_id,
            ))
            t1 = time.perf_counter()
            batch_size, batch = tools.unpack_transitions(resp)
            if batch_size == 0:
                self.logger.debug("download empty data, retry")
                continue
            with lock:
                self.trainer.replay_buffer.put_batch(**batch)
            t2 = time.perf_counter()
            self.logger.debug("downloaded size=%d, request=%.3fs, unpack=%.3fs, reqId='%s'",
                              batch_size, t1 - t0, t2 - t1, req_id)

    def replicate_actor_model(self):
        t0 = time.perf_counter()

        weights_path = os.path.join(self.result_dir, f"params_v{self.version_count}.zip")
        self.trainer.model.save_weights(weights_path)

        req_id = str(uuid4())

        resp = self.buffer_stub.LearnerSetVersion(
            buffer_pb2.LearnerSetVersionReq(
                version=self.version_count,
                requestId=req_id
            ))
        if not resp.done:
            raise ValueError(
                f"buffer set version err: {resp.err}, version={self.version_count}, requestId='{req_id}'")

        futures = {}
        for addr, stub in self.actors_stub.items():
            try:
                resp_future = stub.ReplicateModel.future(
                    tools.read_weights_iterfile(
                        weights_path,
                        version=self.version_count,
                        request_id=req_id
                    ))
                futures[addr] = resp_future
            except grpc.RpcError as e:
                self.logger.error(
                    "actor replicate params err: %e, addr='%s', "
                    "version=%d, requestId='%s'", str(e), addr, self.version_count, req_id)

        for addr, resp_future in futures.items():
            resp = resp_future.result()
            if not resp.done:
                raise ValueError(
                    f"actor replicate params err: {resp.err}, "
                    f"addr='{addr}', version={self.version_count}, requestId='{req_id}'")
        t1 = time.perf_counter()
        self.logger.debug(
            "set buffer and replicate actors params, version=%d, spend=%.3fs, reqId='%s'",
            self.version_count, t1 - t0, req_id)

        self.version_count += 1

    def run(
            self,
            epoch: tp.Optional[int] = None,
            epoch_step: tp.Optional[int] = None,
            download_max_size: int = 1000,
            replicate_step: tp.Optional[int] = None,
    ):
        self.check_actors_buffer_ready()
        self.send_init_data()
        if epoch is None:
            epoch = -1

        lock = threading.Lock()
        td = threading.Thread(target=self.download_data, kwargs=dict(lock=lock, max_size=download_max_size))
        td.start()

        if replicate_step is None and not self.trainer.is_on_policy:
            replicate_step = 100
        count = 0
        epoch_generator = tools.get_count_generator(epoch)
        for ep in epoch_generator:
            while self.trainer.replay_buffer.current_loading_point < 100:
                time.sleep(0.5)
                continue
            if epoch_step is None:
                if self.trainer.replay_buffer.is_full():
                    _ep_step = self.trainer.replay_buffer.max_size // self.trainer.batch_size
                else:
                    _ep_step = self.trainer.replay_buffer.current_loading_point // self.trainer.batch_size
            else:
                _ep_step = epoch_step
            t0 = time.perf_counter()
            for _step in range(_ep_step):
                res = self.trainer.train_batch()
                if self.trainer.is_on_policy:
                    if res.model_replaced:
                        self.replicate_actor_model()
                else:
                    if count % replicate_step == 0:
                        self.replicate_actor_model()
                    count += 1
            t1 = time.perf_counter()
            self.logger.debug("trained %d times in ep=%d, spend=%.2fs", _ep_step, ep, t1 - t0)

        # stop downloading data
        self.keep_download_data = False
        td.join()

        req_id = str(uuid4())
        for addr, stub in self.actors_stub.items():
            resp = stub.Terminate(actor_pb2.TerminateReq(requestId=req_id))
            if not resp.done:
                self.logger.error(
                    "actor not terminated, err: %s, addr='%s', reqId='%s'", resp.err, addr, resp.requestId)
