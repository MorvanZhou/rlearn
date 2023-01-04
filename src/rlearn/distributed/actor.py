import datetime
import os
import shutil
import threading
import typing as tp

import grpc



from rllearn.distributed import actor_gym_env
from rllearn.distributed.buffer_pb2 import UploadDataReq
from rllearn.distributed.buffer_pb2_grpc import ReplayBufferStub
from rllearn.base import BaseRLNet
# from tipesdk.envs.grpc_server import start_grpc_server, EnvService


def _get_wrapped_update(model, buffer_address, update, thread_service: EnvService):
    stub = None
    if buffer_address is not None and not model.is_on_policy:
        channel = grpc.insecure_channel(buffer_address)
        stub = ReplayBufferStub(channel=channel)

    def wrap_update(*args, **kwargs):
        if model.is_on_policy:
            thread_service.try_wait_update()
        res = update(*args, **kwargs)
        if stub is not None and model.replay_buffer.current_loading_point > 100:
            # push data to remote replay buffer
            req = UploadDataReq()
            tipecommon.matrix.pack_transitions(model.replay_buffer, req)
            resp = stub.UploadData(req)
            if not resp.done:
                raise ValueError(f"grpc upload data to buffer err: {resp.err}")
            model.replay_buffer.clear()
        return res

    return wrap_update


def _process(asset_dir, current_time):
    board_path = os.path.abspath(os.path.join(asset_dir, "data", "result", "actor-" + current_time))
    shutil.rmtree(board_path, ignore_errors=True)
    os.makedirs(board_path, exist_ok=True)

    code_path = os.path.join(asset_dir, "code.py")
    user_def_module = tipecommon.dynamic_import(
        path=code_path,
        module_name="user_def",
        variables={
            "train_setting": {
                "board_path": board_path,
            },
            "model_learn": False,  # only collect data but not learn
        }
    )

    try:
        conf = getattr(user_def_module, "conf")
        model = getattr(user_def_module, "model")
        update = getattr(user_def_module, "update")
    except KeyError as e:
        raise ValueError(f"does not found required value in user_def_module: {e}")
    return conf, model, update


def run_gym_env(env_name: str, map_data: tp.Dict[str, tp.Any], asset_dir: str, port=None, buffer_address=None):
    import gym

    env = gym.make(env_name, new_step_api=True, render_mode=None)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
    conf, model, update = _process(asset_dir, current_time)

    grpc_server, thread_service = start_grpc_server(port)
    thread_service.set_train_distributed(model)
    if buffer_address is None:
        jmap = actor_gym_env.ON_POLICY_ENV_JOB_MAP
    else:
        jmap = actor_gym_env.OFF_POLICY_ENV_JOB_MAP

    wrapped_update = _get_wrapped_update(model, buffer_address, update, thread_service)
    td = threading.Thread(target=jmap[env_name], kwargs=dict(
        update=wrapped_update, conf=conf, env=env, map_data=map_data
    ))
    td.start()
    grpc_server.wait_for_termination()
    td.join()


def run_ts_env(env_name: str, map_data: tp.Union[str, tp.Any], asset_dir: str, port=None, buffer_address=None):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
    replay_dir = os.path.abspath(os.path.join(asset_dir, "data", "result", "actor-" + current_time, "replay"))
    shutil.rmtree(replay_dir, ignore_errors=True)
    os.makedirs(replay_dir, exist_ok=True)

    conf, model, update = _process(asset_dir, current_time)
    model: BaseRLNet

    tipesdk.make(
        env=env_name,
        map_data=map_data,
        epoch_setting={
            "Epoch": conf.epochs
        },
        random_seed=1,
        result_dir=replay_dir,
        force=False
    )

    wrapped_update = _get_wrapped_update(model, buffer_address, update)
    tipesdk.hook(update_fn=wrapped_update, port=port, model=model)
    tipesdk.run()


def run(env_name: str, map_data: tp.Union[str, tp.Any], asset_dir: str, port=None, debug=False, buffer_address=None):
    ts_game_envs = set()
    for _env in os.listdir(tipecommon.const.ENV_STAGE_DIR):
        if not os.path.isdir(os.path.join(tipecommon.const.ENV_STAGE_DIR, env_name)):
            continue
        if _env.startswith("__") or _env.startswith("."):
            continue
        ts_game_envs.add(_env)

    if debug:
        tipesdk.log.set_debug_level()
    else:
        tipesdk.log.set_info_level()

    if env_name in ts_game_envs:
        run_ts_env(env_name=env_name, map_data=map_data, asset_dir=asset_dir, port=port, buffer_address=buffer_address)
    else:
        run_gym_env(env_name=env_name, map_data=map_data, asset_dir=asset_dir, port=port, buffer_address=buffer_address)
