import os
import shutil
import tempfile
import unittest

import gymnasium
import numpy as np

import rlearn
from rlearn.trainer import tools


def config_trainer(conf):
    trainer = rlearn.get_trainer_by_name(
        conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), f"test_{conf.trainer}")
    )
    rlearn.set_config_to_trainer(conf, trainer)
    return trainer


class TrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.acrobot_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=2,
            epochs=2,
            action_transform=[0, 1, 2],
            nets=[
                rlearn.NetConfig(
                    input_shape=(6,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 3}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(6,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 3}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            random_network_distillation=rlearn.RandomNetworkDistillationConfig(
                target=rlearn.NetConfig(
                    input_shape=(6,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 3})
                    ]
                ),
                learning_rate=0.01,
            ),
            gamma=0.9,
            learning_rates=(0.01, 0.01),
            replay_buffer=rlearn.ReplayBufferConfig(5),
            replace_step=2,
            not_learn_epochs=0,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )

        self.pendulum_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=2,
            epochs=2,
            action_transform=[[-2, 2]],
            replay_buffer=rlearn.ReplayBufferConfig(5),
            replace_step=2,
            nets=[
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 2}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 2}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            random_network_distillation=rlearn.RandomNetworkDistillationConfig(
                target=rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 2})
                    ]
                ),
                learning_rate=0.01,
            ),
            gamma=0.9,
            learning_rates=(0.01, 0.01),
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )

    def tearDown(self) -> None:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), "tmp"), ignore_errors=True)

    def train_pendulum(self, conf):
        trainer = config_trainer(conf)

        env = gymnasium.make(
            'Pendulum-v1',
        )
        for ep in range(conf.epochs):
            s, _ = env.reset()
            for _ in range(5):  # in one episode
                _a = trainer.predict(s)
                self.assertIsInstance(_a, np.ndarray)
                self.assertEqual(1, len(_a))
                # IMPORTANT: it is better to record permuted action in buffer
                a = trainer.map_action(_a)
                self.assertIsInstance(a, np.ndarray)
                self.assertEqual(1, len(a))
                s_, r, _, _, _ = env.step(a)
                # IMPORTANT: it is better to record permuted action in buffer
                trainer.store_transition(s, _a, r, s_)
                s = s_
                trainer.train_batch()
            if ep == 0:
                dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                trainer.save_model_weights(dir_)
                trainer.load_model_weights(dir_)
            trainer.trace({
                "ep_reward": 0,
            }, step=ep)
        env.close()

    def train_arcobot(self, conf):
        trainer = config_trainer(conf)
        env = gymnasium.make(
            'Acrobot-v1',
        )
        for ep in range(conf.epochs):
            s, _ = env.reset()
            for _ in range(5):  # in one episode
                _a = trainer.predict(s)
                self.assertIsInstance(_a, int)
                # IMPORTANT: it is better to record permuted action in buffer
                a = trainer.map_action(_a)
                self.assertIsInstance(a, int)
                s_, r, _, _, _ = env.step(a)
                # IMPORTANT: it is better to record permuted action in buffer
                trainer.store_transition(s, _a, r, s_)

                s = s_
                trainer.train_batch()

            if ep == 0:
                dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                trainer.save_model_weights(dir_)
                trainer.load_model_weights(dir_)
            trainer.trace({
                "ep_reward": 0,
            }, step=ep)
        env.close()

    def test_all_trainers(self):
        all_trainer_dict = tools.get_all()
        for k, t in all_trainer_dict.items():
            trainer = t()
            self.assertEqual(trainer.model.is_on_policy, trainer.is_on_policy)
            self.assertEqual(k, trainer.name)
            self.assertTrue(k.startswith(trainer.model.name))

            if trainer.model.is_discrete_action:
                self.acrobot_conf.trainer = k
                self.train_arcobot(self.acrobot_conf)
            else:
                self.pendulum_conf.trainer = k
                self.train_pendulum(self.pendulum_conf)
