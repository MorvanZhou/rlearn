import os
import unittest

import numpy as np
from tensorflow import keras

import rlearn
from rlearn.model import tools


class ModelTest(unittest.TestCase):
    def test_model_properties(self):
        ps = {
            "ActorCriticDiscrete": {"is_on_policy": True, "is_discrete_action": True, "predicted_model_name": "actor"},
            "ActorCriticContinue": {"is_on_policy": True, "is_discrete_action": False, "predicted_model_name": "actor"},
            "DQN": {"is_on_policy": False, "is_discrete_action": True, "predicted_model_name": "q"},
            "DDPG": {"is_on_policy": False, "is_discrete_action": False, "predicted_model_name": "actor"},
            "DuelingDQN": {"is_on_policy": False, "is_discrete_action": True, "predicted_model_name": "q"},
            "PPODiscrete": {"is_on_policy": True, "is_discrete_action": True, "predicted_model_name": "pi_"},
            "PPOContinue": {"is_on_policy": True, "is_discrete_action": False, "predicted_model_name": "pi_"},
            "SACDiscrete": {"is_on_policy": False, "is_discrete_action": True, "predicted_model_name": "actor"},
            "SACContinue": {"is_on_policy": False, "is_discrete_action": False, "predicted_model_name": "actor"},
            "TD3": {"is_on_policy": False, "is_discrete_action": False, "predicted_model_name": "actor"},
        }
        for name, m in tools.get_all().items():
            self.assertEqual(ps[name]["is_on_policy"], m.is_on_policy)
            self.assertEqual(ps[name]["is_discrete_action"], m.is_discrete_action)
            self.assertEqual(ps[name]["predicted_model_name"], m.predicted_model_name)
            m_instance = m()
            self.assertEqual(0, len(m_instance.models))

    def test_dqn(self):
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        m = rlearn.DQN()
        m.set_encoder(net, 3)
        self.assertIsInstance(m.predict(np.zeros([2, ])), int)

    def test_dqn_add_model(self):
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(3)
        ])
        m = rlearn.DQN()
        m.set_model(net)
        pred = m.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, int)

    def test_ppo_continuous(self):
        m = rlearn.PPOContinue()
        m.set_encoder(
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
        action_transformer = rlearn.transformer.ContinuousAction([[0, 360]])

        for _ in range(10):
            a = m.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_ppo_continuous_add_model(self):
        m = rlearn.PPOContinue()
        m.set_model(
            pi=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
                keras.layers.Dense(2 * 2),  # mean, stddiv
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
                keras.layers.Dense(1)
            ]),
        )
        a = m.predict(np.random.random((2,)))
        self.assertEqual(2, len(a))

    def test_ppo_discrete(self):
        m = rlearn.PPODiscrete()
        m.set_encoder(
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

        for _ in range(10):
            a = m.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_dueling_dqn(self):
        m = rlearn.DuelingDQN()
        m.set_encoder(keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ]),
            action_num=3
        )
        self.assertIsInstance(m.predict(np.zeros([2, ])), int)
        self.assertEqual(rlearn.DuelingDQN.name, m.name)

    def test_ddpg(self):
        m = rlearn.DDPG()
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=3
        )
        pred = m.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rlearn.DDPG.name, m.name)

    def test_sac_continue(self):
        m = rlearn.SACContinue()
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=3
        )
        pred = m.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rlearn.SACContinue.name, m.name)

    def test_sac_discrete(self):
        m = rlearn.SACDiscrete()
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )

        for _ in range(10):
            a = m.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_ddpg_add_model(self):
        m = rlearn.DDPG()
        m.set_model(
            actor=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
                keras.layers.Dense(2),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
                keras.layers.Dense(1)
            ]),
        )
        a = m.predict(np.random.random((2,)))
        self.assertEqual(2, len(a))

    def test_save_ckpt_model(self):
        m = rlearn.DDPG()
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
            ]),
            action_num=3
        )
        path = "tmp_model.zip"
        m.save_weights(path)
        self.assertTrue(os.path.isfile(path))
        m.load_weights(path)
        os.remove(path)

    def test_save_pb_model(self):
        m = rlearn.DDPG()
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
            ]),
            action_num=3
        )
        path = "tmp_model.zip"
        m.save(path)
        self.assertTrue(os.path.isfile(path))
        m.load(path)
        os.remove(path)

    def test_get_flat_weights(self):
        m = rlearn.DQN()
        m.set_encoder(
            encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
                keras.layers.Dense(5)
            ]),
            action_num=2
        )
        weights = m.get_flat_weights()
        self.assertEqual(82, weights.size)

        m.set_flat_weights(weights)

        m = rlearn.DQN(training=False)
        m.set_encoder(
            encoder=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
                keras.layers.Dense(5)
            ]),
            action_num=2
        )
        weights = m.get_flat_weights()
        self.assertEqual(41, weights.size)

        m.set_flat_weights(weights)

        m = rlearn.DDPG(training=True)
        m.set_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
                keras.layers.Dense(5)
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
                keras.layers.Dense(5)
            ]),
            action_num=1
        )
        weights = m.get_flat_weights()
        self.assertEqual(1506, weights.size)

        m.set_flat_weights(weights)


class RNDTest(unittest.TestCase):
    def test_rnd(self):
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.ReLU(),
            keras.layers.Dense(10),
        ])
        rnd = rlearn.RND(target=net)
        int_r = rnd.intrinsic_reward(np.random.random((3, 2)))
        self.assertEqual((3,), int_r.shape)

        int_r = rnd.intrinsic_reward(np.random.random((1, 2)))
        self.assertEqual((1,), int_r.shape)
