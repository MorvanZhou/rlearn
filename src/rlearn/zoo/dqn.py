from rlearn import k
from rlearn import model


class DQNSmall(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = k.Sequential([
            k.layers.InputLayer(self.state_dim),
            k.layers.Dense(32),
            k.layers.ReLU(),
            k.layers.Dense(32),
            k.layers.ReLU(),
        ])
        self.set_encoder(net, self.action_dim)


class DQNMiddle(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = k.Sequential([
            k.layers.InputLayer(self.state_dim),
            k.layers.Dense(128),
            k.layers.ReLU(),
            k.layers.Dense(128),
            k.layers.ReLU(),
            k.layers.Dense(128),
            k.layers.ReLU(),
            k.layers.Dense(self.action_dim)
        ])
        self.set_encoder(net, self.action_dim)


class DQNLarge(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = k.Sequential([
            k.layers.InputLayer(self.state_dim),
            k.layers.Dense(256),
            k.layers.ReLU(),
            k.layers.Dense(256),
            k.layers.ReLU(),
            k.layers.Dense(256),
            k.layers.ReLU(),
            k.layers.Dense(256),
            k.layers.ReLU(),
            k.layers.Dense(self.action_dim)
        ])
        self.set_encoder(net, self.action_dim)


__all__ = ["DQNLarge", "DQNSmall", "DQNMiddle"]
