import gymnasium
from tensorflow import keras

import rlearn

env = gymnasium.make('CartPole-v1', render_mode="human")
trainer = rlearn.DQNTrainer()
# trainer.set_replay_buffer(1000)
trainer.set_model_encoder(
    q=keras.Sequential([
        keras.layers.InputLayer(4),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ]),
    action_num=env.action_space.n
)

for _ in range(100):
    s, _ = env.reset()
    for _ in range(200):
        a = trainer.predict(s)
        s_, r, done, _, _ = env.step(a)
        trainer.store_transition(s, a, r, s_, done)
        trainer.train_batch()
        s = s_
        if done:
            break
