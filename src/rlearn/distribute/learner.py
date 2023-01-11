import datetime
import logging
import os
import threading
import time
import typing as tp
from uuid import uuid4

import grpc

from rlearn.distribute import logger, actor_pb2_grpc, buffer_pb2_grpc, actor_pb2, buffer_pb2, tools
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
        req_id = str(uuid4())
        for _ in range(10):
            try:
                resp = self.buffer_stub.ServiceReady(buffer_pb2.ServiceReadyReq(requestId=req_id))
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
                    resp = stub.ServiceReady(actor_pb2.ServiceReadyReq(requestId=req_id))
                    if not resp.ready:
                        raise ValueError(f"actor at {addr} not ready: {resp.err}, requestId='{resp.requestId}'")
                    self.logger.debug("connected to actor %s", addr)
                    break
                except grpc.RpcError:
                    self.logger.debug("waiting for actor (%s)", addr)
                    time.sleep(1)

        self.logger.debug("actors server ready, reqId='%s'", req_id)

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
                        model_type=self.trainer.model.name,
                        max_episode=0,
                        max_episode_step=0,
                        version=self.version_count,
                        request_id=req_id,
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
            t0 = time.process_time()
            resp = self.buffer_stub.DownloadData(buffer_pb2.DownloadDataReq(
                maxSize=max_size,
                requestId=req_id,
            ))
            t1 = time.process_time()
            s, a, r, s_ = tools.unpack_transitions(resp)
            if len(s) == 0:
                self.logger.debug("download empty data, retry")
                continue
            with lock:
                self.trainer.replay_buffer.put_batch(s, a, r, s_)
            t2 = time.process_time()
            self.logger.debug("downloaded size=%d, request=%.3fs, unpack=%.3fs, reqId='%s'",
                              len(s), t1 - t0, t2 - t1, req_id)

    def replicate_actor_model(self):
        t0 = time.process_time()

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
        t1 = time.process_time()
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
            t0 = time.process_time()
            for _ in range(_ep_step):
                self.trainer.train_batch()
                if self.trainer.is_on_policy:
                    if self.trainer.model_replaced:
                        self.replicate_actor_model()
                else:
                    if count % replicate_step == 0:
                        self.replicate_actor_model()
                    count += 1
            t1 = time.process_time()
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

# import datetime
# import logging
# import os
# import shutil
# import time
# import typing as tp
# from abc import abstractmethod, ABCMeta
#
# import grpc
#
# from tipealgs.ml import rl
# from tipecommon import game_pb2_grpc, game_pb2, dynamic_import, matrix
# from tipecommon.tools import read_iterfile, get_default_format_logger
#
#
# class Learner(metaclass=ABCMeta):
#     def __init__(
#             self,
#             asset_dir: str,
#             epoch_training_time: float,
#             actors_address: tp.List[str],
#             debug: bool = False,
#     ):
#         self.logger = get_default_format_logger("distLearner")
#         if debug:
#             self.logger.setLevel(logging.DEBUG)
#         else:
#             self.logger.setLevel(logging.ERROR)
#         current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
#         board_path = os.path.abspath(os.path.join(asset_dir, "data", "result", "learner-" + current_time))
#         shutil.rmtree(board_path, ignore_errors=True)
#         code_path = os.path.join(asset_dir, "code.py")
#         user_def_module = dynamic_import(
#             path=code_path,
#             module_name="user_def",
#             variables={
#                 "train_setting": {
#                     "board_path": board_path,
#                 },
#             }
#         )
#
#         try:
#             conf = getattr(user_def_module, "conf")
#             model = getattr(user_def_module, "model")
#         except KeyError as e:
#             raise ValueError(f"does not found required value in user_def_module: {e}")
#
#         self.config: tp.Union[rl.rl_config.QTrainConfig, rl.rl_config.ACTrainConfig] = conf
#         self.actors_channel: tp.Dict[str, grpc.Channel] = {
#             address: grpc.insecure_channel(address) for address in actors_address
#         }
#         self.actors_stub: tp.Dict[str, game_pb2_grpc.EnvStub] = {
#             address: game_pb2_grpc.EnvStub(channel=channel) for address, channel in self.actors_channel.items()
#         }
#         self.model = model
#         self.sync_count = 0
#         self.model_dir = os.path.abspath(
#             os.path.join(asset_dir, "data", "result", "learner-" + current_time, "model"))
#         os.makedirs(self.model_dir, exist_ok=True)
#         self.epoch_training_time = epoch_training_time
#
#     @abstractmethod
#     def download_data(self):
#         pass
#
#     def replicate(self, model_path: str):
#         start_time = time.time()
#         actor_res = {}
#         all_set = True
#         futures = []
#         for address, stub in self.actors_stub.items():
#             try:
#                 resp_future = stub.ReplicateModel.future(read_iterfile(model_path))
#             except grpc.RpcError as e:
#                 self.logger.error(
#                     "replicate model failed, address=%s connection is failed: %s",
#                     address, str(e).replace("\n", "\\n"))
#                 continue
#             futures.append((address, resp_future))
#
#         for address, resp_future in futures:
#             resp = resp_future.result()
#             if not resp.done:
#                 self.logger.error("replicate model failed, address=%s sync not done", address)
#                 actor_res[address] = False
#                 all_set = False
#                 continue
#
#             actor_res[address] = True
#
#         self.sync_count += 1
#         self.logger.debug(
#             "sync_count=%d, replicate time spend=%.3f s",
#             self.sync_count, time.time() - start_time)
#         return all_set, actor_res
#
#     def wait_all_actors_ready(self):
#         done = True
#         for address, stub in self.actors_stub.items():
#             try:
#                 resp = stub.ServiceReady(game_pb2.ServiceReadyReq())
#             except grpc.RpcError:
#                 self.logger.info("wait actor %s to be ready", address)
#                 done = False
#                 break
#             if not resp.ready:
#                 self.logger.info("wait actor %s to be ready", address)
#                 done = False
#                 break
#         if not done:
#             time.sleep(1)
#             return self.wait_all_actors_ready()
#         self.logger.debug("all actors are ready")
#         return done
#
#     @abstractmethod
#     def _run_epoch(self, ep):
#         pass
#
#     def run(self, epoch: int):
#         self.wait_all_actors_ready()
#         final_epoch = epoch
#         if epoch < 1:
#             final_epoch = -1
#         ep = 0
#         while True:
#             if ep == final_epoch:
#                 break
#             self._run_epoch(ep)
#             ep += 1
#
#
# class OnPolicyLearner(Learner):
#     def __init__(
#             self,
#             asset_dir: str,
#             epoch_training_time: float,
#             actors_address: tp.List[str],
#             wait_actor_time: float = 0.2,
#             debug: bool = False,
#     ):
#         super().__init__(
#             asset_dir=asset_dir,
#             epoch_training_time=epoch_training_time,
#             actors_address=actors_address,
#             debug=debug
#         )
#         self.wait_actor_time = wait_actor_time
#
#     def _run_epoch(self, ep):
#         self.download_data()
#         start_time = time.time()
#         while time.time() - start_time < self.epoch_training_time:
#             self.model.learn()
#         model_path = os.path.join(self.model_dir, f"ep{ep}.zip")
#         self.model.save(model_path)
#         self.replicate(model_path)
#         time.sleep(self.wait_actor_time)
#
#     def download_data(self):
#         # from actors
#         start_time = time.time()
#         self.model.replay_buffer.clear()
#         futures = []
#         for address, stub in self.actors_stub.items():
#             try:
#                 resp_future = stub.DownloadData.future(game_pb2.DownloadDataReq())
#             except grpc.RpcError as e:
#                 self.logger.error(
#                     "download data from actor(address=%s) failed, connection err: %s",
#                     address, str(e).replace("\n", "\\n"))
#                 continue
#             futures.append((stub, resp_future))
#
#         for stub, resp_future in futures:
#             resp = resp_future.result()
#             s, a, r, s_ = matrix.unpack_transitions(resp)
#             while len(s) == 0:
#                 self.logger.warning("buffer is empty, auto retry")
#                 time.sleep(1.)
#                 resp = stub.DownloadData(game_pb2.DownloadDataReq())
#                 s, a, r, s_ = matrix.unpack_transitions(resp)
#             self.model.replay_buffer.put_batch(s, a, r, s_)
#         self.logger.debug(
#             "download data from actor, data_size=%d spend=%.3f s",
#             self.model.replay_buffer.current_loading_point,
#             time.time() - start_time
#         )
#
#
# class OffPolicyLearner(Learner):
#     def __init__(
#             self,
#             asset_dir: str,
#             epoch_training_time: float,
#             actors_address: tp.List[str],
#             buffer_address: str,
#             debug: bool = False,
#     ):
#         super().__init__(
#             asset_dir=asset_dir,
#             epoch_training_time=epoch_training_time,
#             actors_address=actors_address,
#             debug=debug
#         )
#         self.buffer_address: str = buffer_address
#         self.buffer_stub = rl.distributed.replaybuf.buffer_pb2_grpc.ReplayBufferStub(
#             channel=grpc.insecure_channel(self.buffer_address)
#         )
#         self.wait_buffer_ready()
#
#     def wait_buffer_ready(self):
#         done = True
#         try:
#             resp = self.buffer_stub.ServiceReady(rl.distributed.replaybuf.buffer_pb2.ServiceReadyReq())
#             if not resp.ready:
#                 self.logger.info("wait buffer %s to be ready", self.buffer_address)
#                 done = False
#         except grpc.RpcError:
#             self.logger.info("wait buffer %s to be ready", self.buffer_address)
#             done = False
#
#         if not done:
#             time.sleep(1)
#             return self.wait_buffer_ready()
#         self.logger.debug("buffer is ready")
#         return done
#
#     def _run_epoch(self, ep):
#         self.download_data()
#         start_time = time.time()
#         while time.time() - start_time < self.epoch_training_time:
#             self.model.learn()
#         model_path = os.path.join(self.model_dir, f"ep{ep}.zip")
#         self.model.save(model_path)
#         self.replicate(model_path)
#
#     def download_data(self):
#         # from buffer
#         start_time = time.time()
#         resp = self.buffer_stub.DownloadData(rl.distributed.replaybuf.buffer_pb2.DownloadDataReq(maxSize=3000))
#         s, a, r, s_ = matrix.unpack_transitions(resp)
#         if len(s) == 0:
#             self.logger.warning("buffer is empty, auto retry")
#             time.sleep(1.)
#             return self.download_data()
#         self.model.replay_buffer.put_batch(s, a, r, s_)
#         self.logger.debug(
#             "download data from remote buffer, data_size=%d spend=%.3f s",
#             len(s),
#             time.time() - start_time
#         )
