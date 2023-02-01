
# Algorithms

- On-Policy
  - A2C: Actor-Critic
  - PPO: Proximal Policy Optimization
- Off-Policy
  - DQN: Deep Q Networks
  - DuelingDQN: Dueling DQN
  - DDPG: Deep Deterministic Policy Gradients
  - TD3: Twin Delayed DDPG
  - SAC: Soft Actor Critic

# Usage

## Classical pattern

```python
import gymnasium
from tensorflow import keras

import rlearn

# define an environment
env = gymnasium.make('CartPole-v1', render_mode="human")

# set reinforcement learning trainer
trainer = rlearn.DQNTrainer()
trainer.set_replay_buffer(max_size=1000)
trainer.set_model_encoder(
  q=keras.Sequential([
    keras.layers.InputLayer(4),  # state has dimension of 4
    keras.layers.Dense(32),
    keras.layers.ReLU(),
  ]),
  action_num=env.action_space.n
)

# training loop
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
```

## Parallel training

Remote buffer:

```python
from rlearn import distributed

distributed.experience.start_replay_buffer_server(
  port=50051,
  max_size=1000,
  buf="RandomReplayBuffer",
)
```

Actors:

```python
from rlearn import distributed
import gymnasium


class CartPole(rlearn.EnvWrapper):
  def __init__(self, render_mode="human"):
    self.env = gymnasium.make('CartPole-v1', render_mode=render_mode)

  def reset(self):
    s, _ = self.env.reset()
    return s

  def step(self, a):
    s_, _, done, _, _ = self.env.step(a)
    r = -1 if done else 0
    return s_, r, done


distributed.experience.start_actor_server(
  port=50052,
  remote_buffer_address="localhost:50051",
  local_buffer_size=10,
  env=CartPole(),
)
```

Learner:

```python
import rlearn
from tensorflow import keras

trainer = rlearn.trainer.DQNTrainer()
trainer.set_model_encoder(
  q=keras.Sequential([
    keras.layers.InputLayer(4),
    keras.layers.Dense(32),
    keras.layers.ReLU(),
  ]),
  action_num=2
)
trainer.set_params(
  learning_rate=0.01,
  batch_size=32,
  replace_step=15,
)
trainer.set_action_transformer(rlearn.transformer.DiscreteAction([0, 1]))
learner = rlearn.distributed.experience.Learner(
  trainer=trainer,
  remote_buffer_address="localhost:50051",
  remote_actors_address=["localhost:50052", ],
)
learner.run(epoch=200)
```

# Install

```shell
git clone https://git.woa.com/TIPE/rlearn.git
cd rlearn

# apple m1 silicon should use conda command:
conda install -c apple tensorflow-deps
########

python3 setup.py install
```
