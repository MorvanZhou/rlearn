import gym
import numpy as np

import rlearn


class CartPoleSmoothReward(rlearn.EnvWrapper):
    def __init__(self, render_mode=None, seed=None):
        self.env = gym.make('CartPole-v1', new_step_api=True, render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        s = self.env.reset(return_info=False)
        return s

    def step(self, a):
        s_, _, done, _, _ = self.env.step(a)
        x, _, theta, _ = s_
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        r = r1 + r2
        return s_, r, done

    def render(self):
        self.env.render()


class CartPoleDiscreteReward(CartPoleSmoothReward):
    def step(self, a):
        s_, _, done, _, _ = self.env.step(a)
        r = -1 if done else 0
        return s_, r, done

    def render(self):
        self.env.render()


class Pendulum(rlearn.EnvWrapper):
    def __init__(self, render_mode=None, seed=None):
        self.env = gym.make('Pendulum-v1', new_step_api=True, render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)
        self.step_count = 0
        self.ep_step = 100

    def reset(self) -> np.ndarray:
        s = self.env.reset(return_info=False)
        self.step_count = 0
        return s

    def step(self, a):
        s_, r, _, _, _ = self.env.step(a)
        r = (r + 8) / 8
        done = self.step_count >= self.ep_step
        self.step_count += 1
        return s_, r, done

    def render(self):
        self.env.render()
