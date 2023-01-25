import os.path
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

import rlearn
from rlearn.trainer.base import BaseTrainer


def get_default_ddpg_trainer():
    trainer = rlearn.trainer.DDPGTrainer()
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
    trainer.set_model_encoder(actor=actor_encoder, critic=critic_encoder, action_num=3)
    trainer.set_params(
        learning_rate=0.01,
        batch_size=32,
    )
    return trainer


class TrainerTest(unittest.TestCase):
    def test_get_trainer_map(self):
        d = {}
        rlearn.trainer.tools._set_trainer_map(BaseTrainer, d)
        for name in ["PPOContinueTrainer", "DuelingDQNTrainer", "DQNTrainer"]:
            self.assertIn(name, d)

    def test_build_from_conf(self):
        conf = rlearn.TrainConfig(
            trainer="DQNTrainer",
            batch_size=32,
            epochs=1000,
            action_transform=[0, 1],
            nets=[rlearn.NetConfig(
                input_shape=(4,),
                layers=[
                    rlearn.LayerConfig("dense", args={"units": 20}),
                    rlearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.9,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(500),
            replace_step=100,
            not_learn_epochs=5,
            epsilon_decay=0.01,
            min_epsilon=0.1,
        )
        trainer = rlearn.trainer.get_trainer_by_name(
            conf.trainer,
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn")
        )
        rlearn.trainer.set_config_to_trainer(conf, trainer)
        self.assertIsNotNone(trainer.model.q_)
        self.assertIsNotNone(trainer.model.q)

    def test_class_name(self):
        for k, v in rlearn.trainer.tools.get_all().items():
            self.assertEqual(k, v.name)

    def test_ddpg_add_encoder_manually(self):
        trainer = rlearn.trainer.DDPGTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_ddpg"))
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(16)
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32)
            ]),
            action_num=1
        )
        trainer.set_params(learning_rate=0.01)
        self.assertEqual((None, 2), trainer.model.actor.input_shape)
        self.assertEqual((None, 1), trainer.model.actor.output_shape)

    def test_ddpg_add_model(self):
        trainer = rlearn.trainer.DDPGTrainer()
        a = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(16),
            keras.layers.Dense(2)
        ])

        class C(keras.Model):
            def __init__(self):
                super().__init__()
                self.l1 = keras.layers.Dense(32)
                self.o = keras.layers.Dense(1)

            def call(self, x):
                # x is [s, a]
                o = tf.concat(x, axis=1)
                o = self.l1(o)
                return self.o(o)

        trainer.set_model(
            actor=a,
            critic=C(),
        )

        self.assertEqual((None, 2), trainer.model.actor.input_shape)
        self.assertEqual((None, 2), trainer.model.actor.output_shape)

        trainer.set_replay_buffer(8)
        trainer.set_params(learning_rate=0.01, batch_size=8)
        for _ in range(8):
            trainer.store_transition(np.random.random(2), np.random.random(2), np.random.random(), np.random.random(2))
        trainer.train_batch()

    def test_dqn_add_model(self):
        trainer = rlearn.trainer.DQNTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn"))
        trainer.set_model(
            q=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(16),
                keras.layers.Dense(3)
            ])
        )
        self.assertEqual((None, 2), trainer.model.q.input_shape)
        self.assertEqual((None, 3), trainer.model.q.output_shape)
        pred = trainer.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, int)

    def test_dqn(self):
        trainer = rlearn.trainer.DQNTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn"))
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.set_model_encoder(net, 3)
        trainer.set_replay_buffer(100)
        replace_ratio = 0.1
        trainer.set_params(
            learning_rate=[0.01, ],
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertEqual(1, trainer.epsilon)
        self.assertIsInstance(trainer.replay_buffer, rlearn.RandomReplayBuffer)
        v = trainer.model.q.trainable_variables[0][0][0].numpy()
        v_ = trainer.model.q_.trainable_variables[0][0][0].numpy()
        replaced = trainer.try_replace_params(source=trainer.model.q, target=trainer.model.q_)
        self.assertTrue(replaced)
        self.assertAlmostEqual(
            trainer.model.q_.trainable_variables[0][0][0].numpy(),
            v_ * (1 - replace_ratio) + v * replace_ratio)
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)

    def test_ppo_continuous(self):
        trainer = rlearn.PPOContinueTrainer()
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_params(
            learning_rate=[0.01, 0.01],
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )
        action_transformer = rlearn.transformer.ContinuousAction([[0, 360]])

        self.assertEqual(1, trainer.epsilon)
        for _ in range(10):
            a = trainer.model.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_ppo_discrete(self):
        trainer = rlearn.PPODiscreteTrainer()
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_replay_buffer(1000)
        trainer.set_params(
            learning_rate=[0.001, 0.001],
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
        trainer = rlearn.trainer.DuelingDQNTrainer()
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.set_model_encoder(net, 3)
        replace_ratio = 0.1
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)
        self.assertEqual(rlearn.DuelingDQN.name, trainer.model.name)

    def test_ddpg(self):
        trainer = get_default_ddpg_trainer()
        pred = trainer.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rlearn.DDPG.name, trainer.model.name)

    def test_save_ckpt_model(self):
        trainer = get_default_ddpg_trainer()
        path = "tmp_model.zip"
        trainer.save_model_weights(path)
        self.assertTrue(os.path.isfile(path))
        trainer.load_model_weights(path)
        os.remove(path)

    def test_save_pb_model(self):
        trainer = get_default_ddpg_trainer()
        path = "tmp_model.zip"
        trainer.save_model(path)
        self.assertTrue(os.path.isfile(path))
        trainer.load_model(path)
        os.remove(path)
