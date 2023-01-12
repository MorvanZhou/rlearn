import unittest

import rlearn


class TransformerTest(unittest.TestCase):
    def test_discrete(self):
        t = rlearn.transformer.DiscreteAction(actions=[2, 4])
        self.assertEqual(2, t.transform(0))
        self.assertEqual(4, t.transform(1))
        with self.assertRaises(IndexError):
            t.transform(3)

    def test_continues(self):
        t = rlearn.transformer.ContinuousAction(bound=[1, 2])
        self.assertEqual(1.5, t(0.))
        self.assertEqual(1.75, t(0.5))
        self.assertEqual(2, t(3))
        self.assertEqual(1, t(-3))