import os.path
import tempfile
import unittest

import numpy as np
from tensorflow import keras

import rllearn
from rllearn.trainer.base import BaseTrainer


class TrainerTest(unittest.TestCase):
    def test_get_trainer_map(self):
        d = {}
        rllearn.trainer.tools._set_trainer_map(BaseTrainer, d)
        for name in ["PPOContinueTrainer", "DuelingDQNTrainer", "DQNTrainer"]:
            self.assertIn(name, d)

    def test_build_from_conf(self):
        conf = rllearn.TrainConfig(
            trainer="DQNTrainer",
            batch_size=32,
            epochs=1000,
            action_transform=[0, 1],
            nets=[rllearn.NetConfig(
                input_shape=(4,),
                layers=[
                    rllearn.LayerConfig("dense", args={"units": 20}),
                    rllearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.9,
            learning_rates=(0.01,),
            replay_buffer=rllearn.ReplayBufferConfig(500),
            replace_step=100,
            not_learn_epochs=5,
            epsilon_decay=0.01,
            min_epsilon=0.1,
        )
        trainer = rllearn.trainer.get_trainer_by_name(
            conf.trainer,
            conf.learning_rates,
            log_dir=os.path.join(tempfile.tempdir, "test_dqn")
        )
        rllearn.trainer.set_config_to_trainer(conf, trainer)
        self.assertIsNotNone(trainer.model.net_)
        self.assertIsNotNone(trainer.model.net)

    def test_class_name(self):
        for k, v in rllearn.trainer.tools.get_all().items():
            self.assertEqual(k, v.name)

    def test_build_manuel(self):
        trainer = rllearn.trainer.DDPGTrainer(
            learning_rates=[0.01, 0.01],
            log_dir=os.path.join(tempfile.tempdir, "test_ddpg"))
        trainer.build_model(
            actor_encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(16)
            ]),
            critic_encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32)
            ]),
            action_num=1
        )
        self.assertEqual((None, 2), trainer.model.actor.input_shape)
        self.assertEqual((None, 1), trainer.model.actor.output_shape)

    def test_dqn(self):
        trainer = rllearn.trainer.DQNTrainer(
            learning_rates=[0.01],
            log_dir=os.path.join(tempfile.tempdir, "test_dqn"))
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.build_model(net, 3)
        trainer.set_replay_buffer(100)
        replace_ratio = 0.1
        trainer.set_params(
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertEqual(1, trainer.epsilon)
        self.assertIsInstance(trainer.replay_buffer, rllearn.RandomReplayBuffer)
        v = trainer.model.net.trainable_variables[0][0][0].numpy()
        v_ = trainer.model.net_.trainable_variables[0][0][0].numpy()
        replaced = trainer.try_replace_params(src=trainer.model.net, target=trainer.model.net_)
        self.assertTrue(replaced)
        self.assertAlmostEqual(
            trainer.model.net_.trainable_variables[0][0][0].numpy(),
            v_ * (1 - replace_ratio) + v * replace_ratio)
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)

    def test_ppo_continuous(self):
        trainer = rllearn.PPOContinueTrainer(
            learning_rates=[0.01, 0.01],
        )
        trainer.build_model(
            pi_encoder=keras.Sequential([
                keras.layers.InputLayer((2, )),
                keras.layers.Dense(10),
            ]),
            critic_encoder=keras.Sequential([
                keras.layers.InputLayer((2, )),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_params(
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )
        action_transformer = rllearn.transformer.ContinuousAction([[0, 360]])

        self.assertEqual(1, trainer.epsilon)
        for _ in range(10):
            a = trainer.model.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_ppo_discrete(self):
        with self.assertRaises(ValueError):
            rllearn.PPODiscreteTrainer(
                learning_rates=[0.001]
            )
        trainer = rllearn.PPODiscreteTrainer(
            learning_rates=[0.001, 0.001]
        )
        trainer.build_model(
            pi_encoder=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic_encoder=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_replay_buffer(1000)
        trainer.set_params(
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )

        for _ in range(10):
            a = trainer.model.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_dueling_dqn(self):
        trainer = rllearn.trainer.DuelingDQNTrainer(learning_rates=[0.01])
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.build_model(net, 3)
        replace_ratio = 0.1
        trainer.set_params(
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)
        self.assertEqual(rllearn.DuelingDQN.name, trainer.model.name)

    def test_ddpg(self):
        with self.assertRaises(ValueError):
            rllearn.DDPGTrainer(
                learning_rates=[0.001]
            )
        trainer = rllearn.trainer.DDPGTrainer(learning_rates=[0.01, 0.01])
        actor_encoder = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        critic_encoder = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.build_model(actor_encoder=actor_encoder, critic_encoder=critic_encoder, action_num=3)
        trainer.set_params(
            batch_size=32,
        )
        pred = trainer.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rllearn.DDPG.name, trainer.model.name)
