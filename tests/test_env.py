import unittest

import numpy as np

import rlearn_envs


class EnvTest(unittest.TestCase):
    def test_flappy_bird(self):
        env = rlearn_envs.get("flappy_bird")
        env.set_show(False)
        for _ in range(3):
            s = env.reset()
            if s is not None:
                self.assertIsInstance(s, np.ndarray)
            while True:
                s, reward, done = env.step(np.random.choice([0, 1], p=[0.92, 0.08], size=(1,)))
                if s is not None:
                    self.assertIsInstance(s, np.ndarray)
                self.assertIsInstance(reward, int)
                self.assertIsInstance(done, bool)
                env.render()
                if done:
                    break
        env.close()

    def test_junior(self):
        env = rlearn_envs.get("junior")
        env.set_show(False)
        for _ in range(3):
            s = env.reset()

            while True:
                s, reward, done = env.step(np.array([0.5]))
                if s is not None:
                    self.assertIsInstance(s, np.ndarray)
                self.assertIsInstance(reward, int)
                self.assertIsInstance(done, bool)
                env.render()
                if done:
                    break
        env.close()

    def test_jumping_dino(self):
        env = rlearn_envs.get("jumping_dino")
        env.set_show(False)
        for _ in range(3):
            s = env.reset()

            while True:
                s, reward, done = env.step(
                    # np.array([0]),
                    np.random.choice([0, 1], p=[0.7, 0.3], size=(1,))
                )
                if s is not None:
                    self.assertIsInstance(s, np.ndarray)
                self.assertIsInstance(reward, int)
                self.assertIsInstance(done, bool)
                env.render()
                if done:
                    break
        env.close()
