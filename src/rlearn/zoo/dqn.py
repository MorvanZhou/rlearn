from tensorflow import keras

from rlearn import model


def smallDQN(state_dim: int, action_dim: int, training: bool = True):
    m = model.DQN(training=training)
    m.set_encoder(
        encoder=keras.Sequential([
            keras.layers.InputLayer(state_dim),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ]),
        action_num=action_dim)
    return m


def middleDQN(state_dim: int, action_dim: int, training: bool = True):
    m = model.DQN(training=training)
    m.set_encoder(
        encoder=keras.Sequential([
            keras.layers.InputLayer(state_dim),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
            keras.layers.Dense(128),
            keras.layers.ReLU(),
        ]),
        action_num=action_dim)
    return m


def largeDQN(state_dim: int, action_dim: int, training: bool = True):
    m = model.DQN(training=training)
    m.set_encoder(
        encoder=keras.Sequential([
            keras.layers.InputLayer(state_dim),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),
        ]),
        action_num=action_dim)
    return m
