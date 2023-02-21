
# Reinforcement learning Algorithms

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

## Classical way

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

set training hyper parameters

```python
import rlearn

trainer = rlearn.DQNTrainer()
trainer.set_params(
  learning_rate=0.01,
  batch_size=32,
  gamma=0.9,
  replace_ratio=1.,
  replace_step=0,
  min_epsilon=0.1,
  epsilon_decay=1e-3,
)
```

## Parallel training

### experience parallel

Start a remote buffer:

```python
from rlearn import distributed

distributed.experience.start_replay_buffer_server(
  port=50051,
)
```

Start actors:

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
  env=CartPole(),
)
```

Start a learner:

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
  actors_address=["localhost:50052", ],
  actor_buffer_size=10,
  remote_buffer_size=1000,
  remote_buffer_type="RandomReplayBuffer",
)
learner.run(max_train_time=100, max_ep_step=-1)
```

### gradient parallel

Start a parameter server

```python
import rlearn

trainer = rlearn.trainer.DQNTrainer()
trainer.set_model_encoder(
  q=keras.Sequential([
    keras.layers.InputLayer(4),
    keras.layers.Dense(20),
    keras.layers.ReLU(),
  ]),
  action_num=2
)
trainer.set_params(
  learning_rate=0.001,
  batch_size=32,
  replace_step=100,
)
trainer.set_action_transformer(rlearn.transformer.DiscreteAction([0, 1]))

rlearn.distributed.gradient.start_param_server(
  port=50051,
  trainer=trainer,
  sync_step=5,
  worker_buffer_type="RandomReplayBuffer",
  worker_buffer_size=3000,
  max_train_time=60,
  # debug=True,
)
```

Start workers

```python
import gymnasium
import rlearn


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


rlearn.distributed.gradient.worker.run(
  env=CartPole(),
  params_server_address="localhost:50051",
  name="worker1",
  # debug=True,
)
```

## Save and reload

Save entire model

```python
import rlearn
from tensorflow import keras
import numpy as np

# define and save a model
trainer = rlearn.DQNTrainer()
trainer.set_model_encoder(
  keras.Sequential([
    keras.layers.InputLayer(2),
    keras.layers.Dense(32),
  ]), action_num=3)
path = "tmp_model0"
trainer.save_model(path)

# reload directory from path
m = rlearn.load_model(path)
action = m.predict(np.random.random((2,)))
```

Save model parameters and reload to a new trainer or new model.

```python
import rlearn
from tensorflow import keras
import numpy as np

# define and save a model
trainer = rlearn.DQNTrainer()
trainer.set_model_encoder(
  keras.Sequential([
    keras.layers.InputLayer(2),
    keras.layers.Dense(32),
  ]), action_num=3)
path = "tmp_model_weights0"
trainer.save_model_weights(path)

# trainer load parameters from path
trainer2 = rlearn.DQNTrainer()
trainer2.set_model_encoder(
  keras.Sequential([
    keras.layers.InputLayer(2),
    keras.layers.Dense(32),
  ]), action_num=3)
trainer2.load_model_weights(path)
action = trainer2.predict(np.random.random((2,)))

# model load parameters
m = rlearn.DQN()
m.set_encoder(encoder=keras.Sequential([
  keras.layers.InputLayer(2),
  keras.layers.Dense(32),
]), action_num=3)
action = m.predict(np.random.random((2,)))
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
