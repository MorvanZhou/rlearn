import logging
import multiprocessing
import multiprocessing as mp
import os
import typing as tp

from rlearn.env.env_wrapper import EnvWrapper
from rlearn.trainer.base import BaseTrainer
from rlearn.trainer.tools import get_trainer_by_name, set_trainer_action_transformer

# linux default is fork, force set to spawn
mp = mp.get_context('spawn')


class MulProcess(mp.Process):
    def __init__(
            self,
            env: EnvWrapper,
            weights_conn: multiprocessing.connection.Connection,
            logger: logging.Logger,
            debug: bool = False,
    ):
        super().__init__()
        self.logger = logger
        self.weights_conn = weights_conn
        self.debug = debug
        self.trainer_type = ""
        self.buffer_type = ""
        self.buffer_size = 0
        self.training_request_id = ""
        self.init_model_pb_path = ""
        self.trainer: tp.Optional[BaseTrainer] = None
        self.env: EnvWrapper = env
        self.max_episode = 0
        self.max_episode_step = 0
        self.action_transform = None

        mgr = mp.Manager()
        self.ns = mgr.Namespace()

        self.ns.version = 0
        self.ns.episode_num = 0
        self.ns.episode_step_num = 0
        self.ns.exit = False
        self.lock = mgr.Lock()
        self.model_loaded = mgr.Event()
        self.model_loaded.clear()

    def init_params(
            self,
            trainer_type: str,
            model_pb_path: str,
            init_version: int,
            request_id: str,
            buffer_size: int,
            buffer_type: str = "RandomReplayBuffer",
            max_episode: int = 0,
            max_episode_step: int = 0,
            action_transform: list = None,
    ):
        self.trainer_type = trainer_type
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.training_request_id = request_id
        self.init_model_pb_path = model_pb_path
        if action_transform is not None:
            self.action_transform = action_transform
        self.ns.version = init_version
        self.max_episode = 0 if max_episode is None else max_episode
        self.max_episode_step = 0 if max_episode_step is None else max_episode_step

    def set_model(self):
        self.trainer = get_trainer_by_name(self.trainer_type)
        self.trainer.set_params(min_epsilon=0.1, epsilon_decay=1e-3)
        self.trainer.set_replay_buffer(max_size=self.buffer_size, buf=self.buffer_type)
        if self.action_transform is not None:
            set_trainer_action_transformer(self.trainer, self.action_transform)

        self.trainer.load_model(self.init_model_pb_path)
        self.logger.debug("initial model is set from path='%s'", self.init_model_pb_path)
        os.remove(self.init_model_pb_path)
        self.model_loaded.set()
