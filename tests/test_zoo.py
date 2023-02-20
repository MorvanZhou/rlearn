import unittest

import rlearn


class ZooTest(unittest.TestCase):
    def test_dqn(self):
        m = rlearn.zoo.smallDQN(4, 3)
        self.assertEqual((None, 4), m.models["q"].input_shape)
        self.assertEqual((None, 3), m.models["q"].output_shape)
        self.assertTrue(m.training)
        self.assertIn("q_", m.models)

    def test_dqn_not_train(self):
        m = rlearn.zoo.middleDQN(4, 3, training=False)
        self.assertEqual((None, 4), m.models["q"].input_shape)
        self.assertEqual((None, 3), m.models["q"].output_shape)
        self.assertFalse(m.training)
        self.assertNotIn("q_", m.models)
