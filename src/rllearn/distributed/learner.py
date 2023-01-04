import datetime
import logging
import os
import shutil
import time
import typing as tp
from abc import abstractmethod, ABCMeta

import grpc

from tipealgs.ml import rl
from tipecommon import game_pb2_grpc, game_pb2, dynamic_import, matrix
from tipecommon.tools import read_iterfile, get_default_format_logger


class Learner(metaclass=ABCMeta):
    def __init__(
            self,
            asset_dir: str,
            epoch_training_time: float,
            actors_address: tp.List[str],
            debug: bool = False,
    ):
        self.logger = get_default_format_logger("distLearner")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        board_path = os.path.abspath(os.path.join(asset_dir, "data", "result", "learner-" + current_time))
        shutil.rmtree(board_path, ignore_errors=True)
        code_path = os.path.join(asset_dir, "code.py")
        user_def_module = dynamic_import(
            path=code_path,
            module_name="user_def",
            variables={
                "train_setting": {
                    "board_path": board_path,
                },
            }
        )

        try:
            conf = getattr(user_def_module, "conf")
            model = getattr(user_def_module, "model")
        except KeyError as e:
            raise ValueError(f"does not found required value in user_def_module: {e}")

        self.config: tp.Union[rl.rl_config.QTrainConfig, rl.rl_config.ACTrainConfig] = conf
        self.actors_channel: tp.Dict[str, grpc.Channel] = {
            address: grpc.insecure_channel(address) for address in actors_address
        }
        self.actors_stub: tp.Dict[str, game_pb2_grpc.EnvStub] = {
            address: game_pb2_grpc.EnvStub(channel=channel) for address, channel in self.actors_channel.items()
        }
        self.model = model
        self.sync_count = 0
        self.model_dir = os.path.abspath(
            os.path.join(asset_dir, "data", "result", "learner-" + current_time, "model"))
        os.makedirs(self.model_dir, exist_ok=True)
        self.epoch_training_time = epoch_training_time

    @abstractmethod
    def download_data(self):
        pass

    def replicate(self, model_path: str):
        start_time = time.time()
        actor_res = {}
        all_set = True
        futures = []
        for address, stub in self.actors_stub.items():
            try:
                resp_future = stub.ReplicateModel.future(read_iterfile(model_path))
            except grpc.RpcError as e:
                self.logger.error(
                    "replicate model failed, address=%s connection is failed: %s",
                    address, str(e).replace("\n", "\\n"))
                continue
            futures.append((address, resp_future))

        for address, resp_future in futures:
            resp = resp_future.result()
            if not resp.done:
                self.logger.error("replicate model failed, address=%s sync not done", address)
                actor_res[address] = False
                all_set = False
                continue

            actor_res[address] = True

        self.sync_count += 1
        self.logger.debug(
            "sync_count=%d, replicate time spend=%.3f s",
            self.sync_count, time.time() - start_time)
        return all_set, actor_res

    def wait_all_actors_ready(self):
        done = True
        for address, stub in self.actors_stub.items():
            try:
                resp = stub.ServiceReady(game_pb2.ServiceReadyReq())
            except grpc.RpcError:
                self.logger.info("wait actor %s to be ready", address)
                done = False
                break
            if not resp.ready:
                self.logger.info("wait actor %s to be ready", address)
                done = False
                break
        if not done:
            time.sleep(1)
            return self.wait_all_actors_ready()
        self.logger.debug("all actors are ready")
        return done

    @abstractmethod
    def _run_epoch(self, ep):
        pass

    def run(self, epoch: int):
        self.wait_all_actors_ready()
        final_epoch = epoch
        if epoch < 1:
            final_epoch = -1
        ep = 0
        while True:
            if ep == final_epoch:
                break
            self._run_epoch(ep)
            ep += 1


class OnPolicyLearner(Learner):
    def __init__(
            self,
            asset_dir: str,
            epoch_training_time: float,
            actors_address: tp.List[str],
            wait_actor_time: float = 0.2,
            debug: bool = False,
    ):
        super().__init__(
            asset_dir=asset_dir,
            epoch_training_time=epoch_training_time,
            actors_address=actors_address,
            debug=debug
        )
        self.wait_actor_time = wait_actor_time

    def _run_epoch(self, ep):
        self.download_data()
        start_time = time.time()
        while time.time() - start_time < self.epoch_training_time:
            self.model.learn()
        model_path = os.path.join(self.model_dir, f"ep{ep}.zip")
        self.model.save(model_path)
        self.replicate(model_path)
        time.sleep(self.wait_actor_time)

    def download_data(self):
        # from actors
        start_time = time.time()
        self.model.replay_buffer.clear()
        futures = []
        for address, stub in self.actors_stub.items():
            try:
                resp_future = stub.DownloadData.future(game_pb2.DownloadDataReq())
            except grpc.RpcError as e:
                self.logger.error(
                    "download data from actor(address=%s) failed, connection err: %s",
                    address, str(e).replace("\n", "\\n"))
                continue
            futures.append((stub, resp_future))

        for stub, resp_future in futures:
            resp = resp_future.result()
            s, a, r, s_ = matrix.unpack_transitions(resp)
            while len(s) == 0:
                self.logger.warning("buffer is empty, auto retry")
                time.sleep(1.)
                resp = stub.DownloadData(game_pb2.DownloadDataReq())
                s, a, r, s_ = matrix.unpack_transitions(resp)
            self.model.replay_buffer.put_batch(s, a, r, s_)
        self.logger.debug(
            "download data from actor, data_size=%d spend=%.3f s",
            self.model.replay_buffer.current_loading_point,
            time.time() - start_time
        )


class OffPolicyLearner(Learner):
    def __init__(
            self,
            asset_dir: str,
            epoch_training_time: float,
            actors_address: tp.List[str],
            buffer_address: str,
            debug: bool = False,
    ):
        super().__init__(
            asset_dir=asset_dir,
            epoch_training_time=epoch_training_time,
            actors_address=actors_address,
            debug=debug
        )
        self.buffer_address: str = buffer_address
        self.buffer_stub = rl.distributed.replaybuf.buffer_pb2_grpc.ReplayBufferStub(
            channel=grpc.insecure_channel(self.buffer_address)
        )
        self.wait_buffer_ready()

    def wait_buffer_ready(self):
        done = True
        try:
            resp = self.buffer_stub.ServiceReady(rl.distributed.replaybuf.buffer_pb2.ServiceReadyReq())
            if not resp.ready:
                self.logger.info("wait buffer %s to be ready", self.buffer_address)
                done = False
        except grpc.RpcError:
            self.logger.info("wait buffer %s to be ready", self.buffer_address)
            done = False

        if not done:
            time.sleep(1)
            return self.wait_buffer_ready()
        self.logger.debug("buffer is ready")
        return done

    def _run_epoch(self, ep):
        self.download_data()
        start_time = time.time()
        while time.time() - start_time < self.epoch_training_time:
            self.model.learn()
        model_path = os.path.join(self.model_dir, f"ep{ep}.zip")
        self.model.save(model_path)
        self.replicate(model_path)

    def download_data(self):
        # from buffer
        start_time = time.time()
        resp = self.buffer_stub.DownloadData(rl.distributed.replaybuf.buffer_pb2.DownloadDataReq(maxSize=3000))
        s, a, r, s_ = matrix.unpack_transitions(resp)
        if len(s) == 0:
            self.logger.warning("buffer is empty, auto retry")
            time.sleep(1.)
            return self.download_data()
        self.model.replay_buffer.put_batch(s, a, r, s_)
        self.logger.debug(
            "download data from remote buffer, data_size=%d spend=%.3f s",
            len(s),
            time.time() - start_time
        )
