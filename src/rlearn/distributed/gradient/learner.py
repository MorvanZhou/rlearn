import logging
import os
import tempfile
import time
import typing as tp
import zlib
from uuid import uuid4

import grpc
import numpy as np

from rlearn.distributed import logger, tools
from rlearn.distributed.gradient import worker_pb2_grpc, worker_pb2
from rlearn.trainer.base import BaseTrainer


class Learner:
    def __init__(
            self,
            trainer: BaseTrainer,
            remote_actors_address: tp.Sequence[str],
            actor_buffer_size: int,
            actor_buffer_type: str = "RandomReplayBuffer",
            debug: bool = False,
    ):
        self.trainer: BaseTrainer = trainer
        self.debug: bool = debug
        self.logger = logger.get_logger("PS")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        self.actor_buffer_size: int = actor_buffer_size
        self.actor_buffer_type: str = actor_buffer_type

        self.actors_channel: tp.Dict[str, grpc.Channel] = {
            address: grpc.insecure_channel(address) for address in remote_actors_address
        }
        self.actors_stub: tp.Dict[str, worker_pb2_grpc.WorkerStub] = {
            address: worker_pb2_grpc.WorkerStub(channel=channel)
            for address, channel in self.actors_channel.items()
        }
        self.version_count = 0

    def check_workers_ready(self):
        timeout = 15
        for addr in self.actors_stub.keys():
            try:
                grpc.channel_ready_future(self.actors_channel[addr]).result(timeout=timeout)
            except grpc.FutureTimeoutError:
                raise ValueError(f"connect actor at {addr} timeout: {timeout}")

        self.logger.debug("actors server ready")

    def send_init_data(self):
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
        self.version_count += 1

    def apply_gradients(self):
        req_id = str(uuid4())
        futures = {}
        for addr, stub in self.actors_stub.items():
            try:
                resp_future = stub.GetGradients.future(worker_pb2.GetGradientsReq(requestId=req_id))
                futures[addr] = resp_future
            except grpc.RpcError as e:
                self.logger.error(
                    "actor get gradients err: %e, addr='%s', "
                    "version=%d, requestId='%s'", str(e), addr, self.version_count, req_id)

        cum_grads = None
        for addr, resp_future in futures.items():
            resp_iter = resp_future.result()
            data = bytearray()
            # version = 0
            # request_id = ""
            for resp in resp_iter:
                if resp.meta.version >= 1:
                    # version = resp.meta.version
                    # request_id = resp.meta.requestId
                    continue
                data.extend(resp.chunkData)
            d_data = zlib.decompress(data)
            gradients = np.frombuffer(d_data, dtype=np.float32)
            if cum_grads is None:
                cum_grads = gradients
            else:
                cum_grads += gradients
        self.trainer.apply_flat_gradients(cum_grads / len(self.actors_stub))

    def replicate_actor_model(self):
        t0 = time.perf_counter()

        weights = self.trainer.model.get_flat_weights()
        req_id = str(uuid4())
        futures = {}
        assert self.version_count >= 1, ValueError(f"self.version_count must >= 1, but got {self.version_count}")
        for addr, stub in self.actors_stub.items():
            try:
                resp_future = stub.ReplicateModel.future(
                    tools.get_iter_values(
                        req=worker_pb2.ReplicateModelReq,
                        meta=worker_pb2.ModelMeta,
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
            "set buffer and replicate actors params, version=%d, spend=%.3fs, reqId='%s'",
            self.version_count, t1 - t0, req_id)

        self.version_count += 1

    def run(
            self,
            sync_time
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

        # stop actors
        req_id = str(uuid4())
        for addr, stub in self.actors_stub.items():
            resp = stub.Terminate(actor_pb2.TerminateReq(requestId=req_id))
            if not resp.done:
                self.logger.error(
                    "actor not terminated, err: %s, addr='%s', reqId='%s'", resp.err, addr, resp.requestId)
        for c in self.actors_channel.values():
            c.close()

        # stop downloading data
        self.keep_download_data = False
        td.join()
