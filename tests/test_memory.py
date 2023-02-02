import unittest

import numpy as np
import tensorflow as tf

import rlearn


class MemoryTest(unittest.TestCase):

    def test_random_put_one(self):
        max_size = 10
        m = rlearn.replaybuf.RandomReplayBuffer(max_size=max_size)
        with self.assertRaises(ValueError) as cm:
            m.sample(1)
        self.assertEqual("replay buffer is empty", str(cm.exception))

        s = np.ones([4, ])
        s_ = np.full([4, ], 2)
        a = 1
        r = 2
        m.put_one(s=s, a=a, r=r, s_=s_)
        self.assertEqual(1, m.pointer)
        self.assertEqual(max_size, len(m.data["s"]))
        self.assertEqual(max_size, len(m.data["a"]))
        self.assertEqual(max_size, len(m.data["r"]))
        m.put_one(s=s, a=1.2, r=r, s_=s_)
        self.assertEqual(2, m.pointer)
        self.assertIsInstance(m.sample(2), dict)
        self.assertEqual(4, len(m.sample(2)))
        self.assertEqual(2, len(m.sample(2)["s"]))
        [m.put_one(s=s, a=1.2, r=r, s_=s_) for _ in range(2)]

        [m.put_one(s=s, a=1.2, r=r, s_=s_) for _ in range(10)]
        self.assertTrue(m.is_full())

    def test_random_batch_put(self):
        max_size = 10
        m = rlearn.replaybuf.RandomReplayBuffer(max_size=max_size)
        with self.assertRaises(ValueError) as cm:
            # different batch size
            m.put_batch(
                s=np.full([22, 3], 1), a=np.full([21, 2], 2), r=np.full([22, ], 3), s_=np.full([22, 3], 4))
        self.assertEqual("batch size of 'a'=21  is not the same as others=22", str(cm.exception))
        # expend dimension
        m.put_batch(
            s=np.full([22, 3], 1), a=np.full([22, ], 2), r=np.full([22, 1], 3), s_=np.full([22, 3], 4))
        self.assertEqual(1, m.data["a"].ndim)

        m.clear()
        self.assertEqual(0, len(m.data))
        m.put_batch(
            s=np.arange(66).reshape([22, 3]), a=np.full([22, 2], 2), r=np.full([22, ], 3),
            s_=np.arange(66).reshape([22, 3]))
        self.assertTrue(m.is_full())
        self.assertEqual(2, m.pointer)

    def test_class_name(self):
        for k, v in rlearn.replaybuf.tools.get_all_buffers().items():
            self.assertEqual(k, v.name)

    def test_prioritized_put(self):
        max_size = 10
        m = rlearn.replaybuf.PrioritizedReplayBuffer(max_size)
        m.put_one(s=np.full([4, ], 1), a=1.2, r=2, s_=np.full([4, ], 2))
        self.assertEqual(1, m.tree.sum)
        m.put_one(s=np.full([4, ], 2), a=1.2, r=2, s_=np.full([4, ], 2))
        self.assertEqual(2, m.tree.sum)
        m.put_one(s=np.full([4, ], 3), a=1.2, r=2, s_=np.full([4, ], 2))
        self.assertEqual(3, m.tree.nodes[0])
        self.assertEqual(2, m.tree.nodes[1])
        self.assertEqual(1, m.tree.nodes[2])
        _ = m.sample(3)
        m.try_weighting_loss(tf.convert_to_tensor([0.1, 0.2, 0.3]), tf.convert_to_tensor([0., 0., 0.]))
        self.assertAlmostEqual(np.power(np.array([0.1, 0.2, 0.3]) + m.epsilon, m.alpha).sum(), m.tree.nodes[0])
