import unittest

import numpy as np

import rlearn


class MemoryTest(unittest.TestCase):

    def test_random_put_one(self):
        max_size = 10
        m = rlearn.replaybuf.RandomReplayBuffer(max_size=max_size)
        with self.assertRaises(ValueError):
            m.sample(1)

        s = np.ones([4, ])
        s_ = np.full([4, ], 2)
        a = np.ones([3, ])
        r = 2
        m.put_one(s, a, r, s_)
        self.assertEqual(1, m.pointer)
        self.assertEqual(max_size, len(m.s))
        self.assertEqual(max_size, len(m.a))
        self.assertEqual(max_size, len(m.r))
        m.put_one(s, 1.2, r, s_)
        self.assertEqual(2, m.pointer)
        self.assertEqual(4, len(m.sample(2)))
        self.assertEqual(2, len(m.sample(2)[0]))
        [m.put_one(s, 1.2, r, s_) for _ in range(2)]

        [m.put_one(s, 1.2, r, s_) for _ in range(10)]
        self.assertTrue(m.is_full())

    def test_random_batch_put(self):
        max_size = 10
        m = rlearn.replaybuf.RandomReplayBuffer(max_size=max_size)
        with self.assertRaises(ValueError):
            # different batch size
            m.put_batch(np.full([22, 3], 1), np.full([21, 2], 2), np.full([22, ], 3), np.full([22, 3], 4))
        with self.assertRaises(ValueError):
            # different state shape
            m.put_batch(np.full([22, 4], 1), np.full([22, 2], 2), np.full([22, ], 3), np.full([22, 3], 4))
        with self.assertRaises(ValueError):
            # reward dimension
            m.put_batch(np.full([22, 3], 1), np.full([22, 2], 2), np.full([22, 3], 3), np.full([22, 3], 4))
        with self.assertRaises(ValueError):
            # action dimension
            m.put_batch(np.full([22, 3], 1), np.full([22, ], 2), np.full([22, 1], 3), np.full([22, 3], 4))
        m.put_batch(np.arange(66).reshape([22, 3]), np.full([22, 2], 2), np.full([22, ], 3),
                    np.arange(66).reshape([22, 3]))
        self.assertTrue(m.is_full())
        self.assertEqual(2, m.pointer)

    def test_class_name(self):
        for k, v in rlearn.replaybuf.tools.get_all_buffers().items():
            self.assertEqual(k, v.name)

    def test_prioritized_put(self):
        max_size = 10
        m = rlearn.replaybuf.PrioritizedReplayBuffer(max_size)
        m.put_one(np.full([4, ], 1), 1.2, 2, np.full([4, ], 2))
        self.assertEqual(1, m.tree.sum)
        m.put_one(np.full([4, ], 2), 1.2, 2, np.full([4, ], 2))
        self.assertEqual(2, m.tree.sum)
        m.put_one(np.full([4, ], 3), 1.2, 2, np.full([4, ], 2))
        self.assertEqual(3, m.tree.nodes[0])
        self.assertEqual(2, m.tree.nodes[1])
        self.assertEqual(1, m.tree.nodes[2])
        _ = m.sample(3)
        abs_error = np.array([0.1, 0.2, 0.3])
        m.batch_update(abs_error)
        self.assertAlmostEqual(np.power(abs_error, m.alpha).sum(), m.tree.nodes[0])
