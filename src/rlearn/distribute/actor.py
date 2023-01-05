import os
import logging
import tempfile
import typing as tp
import uuid
import time
from abc import ABC, abstractmethod
import multiprocessing as mp

import numpy as np
import grpc

from rlearn.distribute import actor_pb2, actor_pb2_grpc, buffer_pb2_grpc, buffer_pb2, tools
from rlearn.distribute.logger import get_logger
from rlearn.model.base import BaseRLModel
from rlearn.replaybuf.base import BaseReplayBuffer
from rlearn.transformer import BaseTransformer


class RLEnv(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, a) -> tp.Tuple[np.ndarray, float, bool]:
        # return (state, reward, done)
        pass


class ActorProcess(mp.Process):
    def __init__(
            self,
            model: BaseRLModel,
            buffer: BaseReplayBuffer,
            env: RLEnv,
            remote_buffer_address: str,
            action_transformer: tp.Optional[BaseTransformer] = None,
            max_episode: tp.Optional[int] = None,
            max_episode_step: tp.Optional[int] = None,
            debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.logger = None
        self.model: BaseRLModel = model
        self.buffer: BaseReplayBuffer = buffer
        self.env: RLEnv = env
        self.action_transformer = action_transformer
        self.max_episode = 0 if max_episode is None else max_episode
        self.max_episode_step = 0 if max_episode_step is None else max_episode_step
        self.remote_buffer_address = remote_buffer_address

        mgr = mp.Manager()
        self.ns = mgr.Namespace()
        self.ns.new_model_path = ""
        self.ns.episode_num = 0
        self.ns.episode_step_num = 0
        self.ns.version = ""
        self.lock = mgr.Lock()

    def try_replicate_model(self):
        with self.lock:
            if self.ns.new_model_path != "":
                self.model.load_weights(self.ns.new_model_path)
                self.ns.new_model_path = ""
                self.logger.debug("model parameters replaced")

    @staticmethod
    def get_count_generator(max_count):
        def unlimited_count_generator():
            c = 0
            while True:
                yield c
                c += 1

        def fix_count_generator(max_step: int):
            for c in range(max_step):
                yield c

        if max_count <= 0:
            g = unlimited_count_generator()
        else:
            g = fix_count_generator(max_count)
        return g

    def send_data_to_remote_buffer(self, buf_stub):
        req = buffer_pb2.UploadDataReq()
        tools.pack_transitions(self.buffer, req)
        resp = buf_stub.UploadData(req)
        if not resp.done:
            raise ValueError(f"grpc upload data to buffer err: {resp.err}")
        self.buffer.clear()

    def run(self):
        episode_generator = self.get_count_generator(self.max_episode)
        self.logger = get_logger("ap")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)
        channel = grpc.insecure_channel(self.remote_buffer_address)
        buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        self.ns.episode_num = 0
        for ep in episode_generator:
            self.logger.debug("ep %d", ep)
            self.ns.episode_num = ep
            s = self.env.reset()

            episode_step_generator = self.get_count_generator(self.max_episode_step)
            for step in episode_step_generator:
                self.logger.debug("step %d", step)
                self.ns.episode_step_num = step

                self.try_replicate_model()

                _a = self.model.predict(s)
                if self.action_transformer is not None:
                    a = self.action_transformer.transform(_a)
                else:
                    a = _a
                s_, r, done = self.env.step(a)
                self.buffer.put(s, _a, r, s_)
                if self.buffer.is_full():
                    self.send_data_to_remote_buffer(buf_stub)
                if done:
                    break


class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(self, actor: ActorProcess, buffer_address: str, debug=False):
        self.actor = actor
        self.logger = get_logger("actor")
        self.logger.setLevel(logging.DEBUG if debug else logging.ERROR)

        channel = grpc.insecure_channel(buffer_address)
        self.buffer_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)
        trail_count = 0
        while True:
            if trail_count > 60:
                raise TimeoutError("remote replay buffer connection timeout")
            resp = self.buffer_stub.ServiceReady(buffer_pb2.ServiceReadyReq(requestId=str(uuid.uuid4())))
            if resp.ready:
                break
            self.logger.info("waiting for remote replay buffer connection")
            time.sleep(1)
            trail_count += 1

    def ServiceReady(self, request, context):
        self.logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return actor_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request, context):
        self.logger.debug("""Start | {"reqId": "%s"}""", request.requestId)
        self.actor.start()
        return actor_pb2.StartResp(done=True, err="", requestId=request.requestId)

    def ReplicateModel(self, request, context):
        data = bytearray()
        filepath = ""
        tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.makedirs(tmp_dir)
        for req in request:
            if req.meta:
                filepath = os.path.join(tmp_dir, req.meta.filename)
                with self.actor.lock:
                    self.actor.ns.version = req.meta.version
                self.logger.debug(
                    """ReplicateModel | {"reqId": "%s", "version": "%s"}""",
                    req.meta.requestId, req.meta.version)
                continue
            data.extend(req.chunkData)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(data)

        done = True
        err = ""
        try:
            self.actor.model.load_weights(filepath)
        except FileNotFoundError as e:
            done = False
            err = str(e)
        finally:
            resp = actor_pb2.ReplicateModelResp(done=done, err=err, requestId=request.requestId)
            os.remove(filepath)
        return resp

    def Terminate(self, request, context):
        self.logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.actor.terminate()
        return actor_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


# def start_actor_server(
#         port,
#         buffer_address: str,
#         debug: bool = False
# ) -> tp.Tuple[grpc.Server, ActorService]:
#     server = grpc.server(
#         futures.ThreadPoolExecutor(
#             max_workers=1,  # one for update, one for replicate
#             thread_name_prefix="s"
#         ), options=[
#             ('grpc.max_send_message_length', 1024 * 1024 * 20),
#         ]
#     )
#     actor = Actor(
#         model: BaseRLModel,
#         buffer: BaseReplayBuffer,
#         env: RLEnv,
#         action_transformer: tp.Optional[BaseTransformer] = None,
#         episode_max_step: tp.Optional[int] = None
#     )
#     service = ActorService(max_size, buf, debug)
#     buffer_pb2_grpc.add_ReplayBufferServicer_to_server(service, server)
#     server.add_insecure_port(f'[::]:{port}')
#     server.start()
#     service.logger.info("python grpc has started at http://127.0.0.1:%s", port)
#     return server


# def _get_wrapped_update(model, buffer_address, update, thread_service: EnvService):
#     stub = None
#     if buffer_address is not None and not model.is_on_policy:
#         channel = grpc.insecure_channel(buffer_address)
#         stub = ReplayBufferStub(channel=channel)
#
#     def wrap_update(*args, **kwargs):
#         if model.is_on_policy:
#             thread_service.try_wait_update()
#         res = update(*args, **kwargs)
#         if stub is not None and model.replay_buffer.current_loading_point > 100:
#             # push data to remote replay buffer
#             req = UploadDataReq()
#             tipecommon.matrix.pack_transitions(model.replay_buffer, req)
#             resp = stub.UploadData(req)
#             if not resp.done:
#                 raise ValueError(f"grpc upload data to buffer err: {resp.err}")
#             model.replay_buffer.clear()
#         return res
#
#     return wrap_update
#
#
# def _process(asset_dir, current_time):
#     board_path = os.path.abspath(os.path.join(asset_dir, "data", "result", "actor-" + current_time))
#     shutil.rmtree(board_path, ignore_errors=True)
#     os.makedirs(board_path, exist_ok=True)
#
#     code_path = os.path.join(asset_dir, "code.py")
#     user_def_module = tipecommon.dynamic_import(
#         path=code_path,
#         module_name="user_def",
#         variables={
#             "train_setting": {
#                 "board_path": board_path,
#             },
#             "model_learn": False,  # only collect data but not learn
#         }
#     )
#
#     try:
#         conf = getattr(user_def_module, "conf")
#         model = getattr(user_def_module, "model")
#         update = getattr(user_def_module, "update")
#     except KeyError as e:
#         raise ValueError(f"does not found required value in user_def_module: {e}")
#     return conf, model, update
#
#
# def run_gym_env(env_name: str, map_data: tp.Dict[str, tp.Any], asset_dir: str, port=None, buffer_address=None):
#     import gym
#
#     env = gym.make(env_name, new_step_api=True, render_mode=None)
#
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
#     conf, model, update = _process(asset_dir, current_time)
#
#     grpc_server, thread_service = start_grpc_server(port)
#     thread_service.set_train_distributed(model)
#     if buffer_address is None:
#         jmap = actor_gym_env.ON_POLICY_ENV_JOB_MAP
#     else:
#         jmap = actor_gym_env.OFF_POLICY_ENV_JOB_MAP
#
#     wrapped_update = _get_wrapped_update(model, buffer_address, update, thread_service)
#     td = threading.Thread(target=jmap[env_name], kwargs=dict(
#         update=wrapped_update, conf=conf, env=env, map_data=map_data
#     ))
#     td.start()
#     grpc_server.wait_for_termination()
#     td.join()
#
#
# def run_ts_env(env_name: str, map_data: tp.Union[str, tp.Any], asset_dir: str, port=None, buffer_address=None):
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
#     replay_dir = os.path.abspath(os.path.join(asset_dir, "data", "result", "actor-" + current_time, "replay"))
#     shutil.rmtree(replay_dir, ignore_errors=True)
#     os.makedirs(replay_dir, exist_ok=True)
#
#     conf, model, update = _process(asset_dir, current_time)
#     model: BaseRLModel
#
#     tipesdk.make(
#         env=env_name,
#         map_data=map_data,
#         epoch_setting={
#             "Epoch": conf.epochs
#         },
#         random_seed=1,
#         result_dir=replay_dir,
#         force=False
#     )
#
#     wrapped_update = _get_wrapped_update(model, buffer_address, update)
#     tipesdk.hook(update_fn=wrapped_update, port=port, model=model)
#     tipesdk.run()
#
#
# def run(env_name: str, map_data: tp.Union[str, tp.Any], asset_dir: str, port=None, debug=False, buffer_address=None):
#     ts_game_envs = set()
#     for _env in os.listdir(tipecommon.const.ENV_STAGE_DIR):
#         if not os.path.isdir(os.path.join(tipecommon.const.ENV_STAGE_DIR, env_name)):
#             continue
#         if _env.startswith("__") or _env.startswith("."):
#             continue
#         ts_game_envs.add(_env)
#
#     if debug:
#         tipesdk.log.set_debug_level()
#     else:
#         tipesdk.log.set_info_level()
#
#     if env_name in ts_game_envs:
#         run_ts_env(
#           env_name=env_name, map_data=map_data, asset_dir=asset_dir,
#           port=port, buffer_address=buffer_address)
#     else:
#         run_gym_env(
#               env_name=env_name, map_data=map_data,
#               asset_dir=asset_dir, port=port, buffer_address=buffer_address)
