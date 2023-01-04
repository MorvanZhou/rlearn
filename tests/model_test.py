import unittest

import numpy as np
from tensorflow import keras

import rllearn


class ModelTest(unittest.TestCase):
    def test_dqn(self):
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        m = rllearn.DQN()
        m.build(net, 3)
        self.assertIsInstance(m.predict(np.zeros([2, ])), int)

    def test_ppo_continuous(self):
        m = rllearn.PPOContinue()
        m.build(
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
        action_transformer = rllearn.transformer.ContinuousAction([[0, 360]])

        for _ in range(10):
            a = m.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_ppo_discrete(self):
        m = rllearn.PPODiscrete()
        m.build(
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

        for _ in range(10):
            a = m.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_dueling_dqn(self):
        m = rllearn.DuelingDQN()
        m.build(keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=3
        )
        self.assertIsInstance(m.predict(np.zeros([2, ])), int)
        self.assertEqual(rllearn.DuelingDQN.name, m.name)

    def test_ddpg(self):
        m = rllearn.DDPG()
        m.build(
            actor_encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            critic_encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=3
        )
        pred = m.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rllearn.DDPG.name, m.name)
