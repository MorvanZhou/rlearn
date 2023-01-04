import unittest

import rllearn


class ZooTest(unittest.TestCase):
    def test_dqn(self):
        m = rllearn.zoo.DQNSmall(4, 3)
        self.assertEqual(4, m.state_dim)
        self.assertEqual(3, m.action_dim)
        self.assertTrue(m.training)
        self.assertIsNotNone(m.net_)

    def test_dqn_not_train(self):
        m = rllearn.zoo.DQNMiddle(4, 3, training=False)
        self.assertEqual(4, m.state_dim)
        self.assertEqual(3, m.action_dim)
        self.assertFalse(m.training)
        self.assertIsNone(m.net_)
