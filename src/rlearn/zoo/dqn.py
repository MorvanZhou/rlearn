from tensorflow import keras

from rlearn import model


class DQNSmall(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = keras.Sequential([
            keras.layers.InputLayer(self.state_dim),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        self.build(net, self.action_dim)


class DQNMiddle(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = keras.Sequential([
            keras.layers.InputLayer(self.state_dim),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
            keras.layers.Dense(self.action_dim)
        ])
        self.build(net, self.action_dim)


class DQNLarge(model.DQN):
    def __init__(self, state_dim: int, action_dim: int, training: bool = True):
        super().__init__(training=training)
        self.state_dim = state_dim
        self.action_dim = action_dim
        net = keras.Sequential([
            keras.layers.InputLayer(self.state_dim),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(self.action_dim)
        ])
        self.build(net, self.action_dim)


__all__ = ["DQNLarge", "DQNSmall", "DQNMiddle"]
