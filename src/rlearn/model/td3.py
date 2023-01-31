"""
Twin Delayed DDPG
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values,
which then leads to the policy breaking,
because it exploits the errors in the Q-function.
Twin Delayed DDPG (TD3) is an algorithm that addresses this issue by introducing three critical tricks:

Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”),
and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently
than the Q-function. The paper recommends one policy update for every two Q-function updates.

Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder
for the policy to exploit Q-function errors by smoothing out Q along changes in action.
"""

import numpy as np
from tensorflow import keras

from rlearn.model.ddpg import DDPG


class TD3(DDPG):
    name = __qualname__
    predicted_model_name = "actor"

    def __init__(
            self,
            training: bool = True,
    ):
        super().__init__(training=training)

    def set_encoder(self, actor: keras.Model, critic: keras.Model, action_num: int):
        a = self.set_actor_encoder_callback(actor, action_num)
        c = None
        if self.training:
            c = self.set_critic_encoder_callback(critic, action_num)
        self.set_model(a, c)

    def set_model(self, actor: keras.Model, critic: keras.Model):
        self.models["actor"] = actor
        if self.training:
            self.models["actor_"] = self.clone_model(self.models["actor"])
            self.models["c1"] = critic
            self.models["c1_"] = self.clone_model(self.models["c1"])
            self.models["c2"] = critic
            self.models["c2_"] = self.clone_model(self.models["c2"])

    def predict(self, s) -> np.ndarray:
        a = self.models[self.predicted_model_name].predict(np.expand_dims(s, axis=0), verbose=0).ravel()
        if np.isnan(a).any():
            raise ValueError("action contains NaN")
        return a
