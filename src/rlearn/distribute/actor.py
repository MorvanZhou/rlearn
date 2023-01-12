import logging
import multiprocessing as mp
import os
import random
import string
import tempfile
import time
import typing as tp
import uuid
from concurrent import futures

import grpc

from rlearn.distribute import actor_pb2, actor_pb2_grpc, buffer_pb2_grpc, buffer_pb2, tools
from rlearn.distribute.logger import get_logger
from rlearn.env_wrapper import EnvWrapper
from rlearn.model.base import BaseRLModel
from rlearn.model.tools import get_model_by_name
from rlearn.replaybuf import RandomReplayBuffer
from rlearn.replaybuf.base import BaseReplayBuffer
from rlearn.transformer import BaseTransformer

# linus default is fork, force set to spawn
mp = mp.get_context('spawn')


class ActorProcess(mp.Process):
    def __init__(
            self,
            buffer: BaseReplayBuffer,
            env: EnvWrapper,
            remote_buffer_address: tp.Optional[str] = None,
            action_transformer: tp.Optional[BaseTransformer] = None,
            debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.logger = None
        self.logger_name = ""
        self.model_type = ""
        self.init_model_pb_path = ""
        self.model: tp.Optional[BaseRLModel] = None
        self.buffer: BaseReplayBuffer = buffer
        self.env: EnvWrapper = env
        self.action_transformer = action_transformer
        self.max_episode = 0
        self.max_episode_step = 0
        self.buffer_address = remote_buffer_address
        self.training_request_id = ""

        mgr = mp.Manager()
        self.ns = mgr.Namespace()

        self.ns.new_model_path = ""
        self.ns.episode_num = 0
        self.ns.episode_step_num = 0
        self.ns.version = 0
        self.lock = mgr.Lock()
        self.model_loaded = mgr.Event()
        self.model_loaded.clear()

    def init_params(
            self,
            model_type: str,
            model_pb_path: str,
            init_version: int,
            request_id: str,
            max_episode: int = 0,
            max_episode_step: int = 0,
    ):
        self.model_type = model_type
        self.init_model_pb_path = model_pb_path
        self.training_request_id = request_id
        self.ns.version = init_version
        self.max_episode = 0 if max_episode is None else max_episode
        self.max_episode_step = 0 if max_episode_step is None else max_episode_step

    def set_model(self):
        self.model = get_model_by_name(self.model_type, training=False)
        self.model.load(self.init_model_pb_path)
        self.logger.debug("initial model is set from path='%s'", self.init_model_pb_path)
        os.remove(self.init_model_pb_path)
        self.model_loaded.set()

    def set_model_replicate(self, version: int, weights_path: str):
        with self.lock:
            self.ns.new_model_path = weights_path
            self.ns.version = version
            self.model_loaded.clear()

    def try_replicate_model(self):
        with self.lock:
            if self.ns.new_model_path != "":
                self.model.load_weights(self.ns.new_model_path)
                os.remove(self.ns.new_model_path)
                self.ns.new_model_path = ""
                self.model_loaded.set()
                self.logger.debug("model parameters replaced, version=%d", self.ns.version)

    def send_data_to_remote_buffer(self, buf_stub):
        req = buffer_pb2.UploadDataReq(version=self.ns.version, requestId=self.training_request_id)
        tools.pack_transitions(self.buffer, req)
        resp = buf_stub.UploadData(req)
        if not resp.done:
            raise ValueError(f"grpc upload data to buffer err: {resp.err}")

    def run(self):
        self.logger = get_logger(self.logger_name)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.ERROR)

        episode_generator = tools.get_count_generator(self.max_episode)

        if self.buffer_address is None or self.buffer_address.strip() == "":
            buf_stub = None
        else:
            channel = grpc.insecure_channel(self.buffer_address)
            buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        self.set_model()

        with self.lock:
            self.ns.episode_num = 0

        for ep in episode_generator:
            with self.lock:
                self.ns.episode_num = ep
            s = self.env.reset()

            step = 0
            episode_step_generator = tools.get_count_generator(self.max_episode_step)
            for step in episode_step_generator:
                with self.lock:
                    self.ns.episode_step_num = step

                self.try_replicate_model()

                _a = self.model.predict(s)
                if self.action_transformer is not None:
                    a = self.action_transformer.transform(_a)
                else:
                    a = _a
                s_, r, done = self.env.step(a)
                self.buffer.put(s, _a, r, s_)
                s = s_
                if self.buffer.is_full() and buf_stub is not None:
                    self.send_data_to_remote_buffer(buf_stub)
                    self.buffer.clear()
                if done:
                    break

            self.logger.debug("ep=%d, total_step=%d", ep, step)


class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(self, actor: ActorProcess, debug=False):
        self.actor = actor
        name = "".join([random.choice(string.ascii_lowercase) for _ in range(4)])
        self.actor.logger_name = f"actorP-{name}"
        self.logger = get_logger(f"actor-{name}")
        self.logger.setLevel(logging.DEBUG if debug else logging.ERROR)

        trail_count = 0
        err = ""
        while True:
            if trail_count > 10:
                raise TimeoutError(f"remote replay buffer connection timeout: {err}")
            try:
                channel = grpc.insecure_channel(self.actor.buffer_address)
                buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)
                resp = buf_stub.ServiceReady(buffer_pb2.ServiceReadyReq(requestId=str(uuid.uuid4())))
                if resp.ready:
                    break
            except grpc.RpcError as e:
                err = str(e)
                self.logger.info("waiting for remote replay buffer (%s)", self.actor.buffer_address)
                time.sleep(1)
                trail_count += 1
        self.logger.debug("connected to buffer %s", self.actor.buffer_address)
        channel.close()

    def ServiceReady(self, request, context):
        self.logger.debug("""ServiceReady | {"reqId": "%s"}""", request.requestId)
        return actor_pb2.ServiceReadyResp(ready=True, requestId=request.requestId)

    def Start(self, request_iterator, context):
        data = bytearray()
        filepath = ""
        request_id = ""
        tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        os.makedirs(tmp_dir)
        for req in request_iterator:
            if req.meta.filename != "":
                filepath = os.path.normpath(os.path.join(tmp_dir, req.meta.filename))
                request_id = req.meta.requestId
                self.actor.init_params(
                    model_type=req.meta.modelType,
                    model_pb_path=filepath,
                    init_version=req.meta.version,
                    request_id=request_id,
                    max_episode=req.meta.maxEpisode,
                    max_episode_step=req.meta.maxEpisodeStep,
                )
                self.logger.debug(
                    'Start | '
                    '{"reqId": "%s", "modelType": "%s","filename": "%s", "version": %d, '
                    '"maxEpisode": %d, "maxEpisodeStep": %d}',
                    req.meta.requestId,
                    req.meta.modelType,
                    req.meta.filename,
                    req.meta.version,
                    req.meta.maxEpisode,
                    req.meta.maxEpisodeStep)
                continue
            data.extend(req.chunkData)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(data)

        self.actor.start()
        if self.actor.model_loaded.wait(5):
            done = True
            err = ""
        else:
            done = False
            err = "model load timeout"
        return actor_pb2.StartResp(done=done, err=err, requestId=request_id)

    def ReplicateModel(self, request_iterator, context):
        data = bytearray()
        filepath = ""
        version = 0
        request_id = ""
        tmp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        for req in request_iterator:
            if req.meta.filename != "":
                filepath = os.path.join(tmp_dir, req.meta.filename)
                version = req.meta.version
                request_id = req.meta.requestId
                self.logger.debug(
                    """ReplicateModel | {"reqId": "%s", "version": %d, "filename": "%s"}""",
                    req.meta.requestId, req.meta.version, req.meta.filename)
                continue
            data.extend(req.chunkData)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(data)

        self.actor.set_model_replicate(version, filepath)
        if self.actor.model_loaded.wait(5):
            done = True
            err = ""
        else:
            done = False
            err = "model replicating timeout"
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass

        resp = actor_pb2.ReplicateModelResp(done=done, err=err, requestId=request_id)
        return resp

    def Terminate(self, request, context):
        self.logger.debug("""Terminate | {"reqId": "%s"}""", request.requestId)
        self.actor.terminate()
        return actor_pb2.TerminateResp(done=True, err="", requestId=request.requestId)


def _start_server(
        port: int,
        remote_buffer_address: str,
        local_buffer_size: int,
        env: EnvWrapper,
        action_transformer: tp.Optional[BaseTransformer] = None,
        debug: bool = False
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=1,  # one for update, one for replicate
            thread_name_prefix="distActor"
        ), options=[
            ('grpc.max_send_message_length', 1024 * 1024 * 20),
        ]
    )

    actor_p = ActorProcess(
        buffer=RandomReplayBuffer(max_size=local_buffer_size),
        env=env,
        remote_buffer_address=remote_buffer_address,
        action_transformer=action_transformer,
        debug=debug,
    )
    service = ActorService(
        actor=actor_p,
        debug=debug
    )
    actor_pb2_grpc.add_ActorServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    service.logger.info("actor has started at http://localhost:%d", port)
    return server


def start_actor_server(
        port: int,
        remote_buffer_address: str,
        local_buffer_size: int,
        env: EnvWrapper,
        action_transformer: tp.Optional[BaseTransformer] = None,
        debug: bool = False
) -> grpc.Server:
    server = _start_server(port, remote_buffer_address, local_buffer_size, env, action_transformer, debug)
    server.wait_for_termination()
    return server

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
