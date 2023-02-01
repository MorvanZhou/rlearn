import multiprocessing
import os
import shutil
import tempfile
import unittest

import gymnasium
from tensorflow import keras

import gym_wrapper_test
import rlearn
from rlearn.distributed import tools


def cartpole_reward(s_, env):
    x, _, theta, _ = s_
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    r = r1 + r2
    return r


def train_cartpole(conf, render_mode="human"):
    trainer = rlearn.get_trainer_by_name(
        conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), f"test_{conf.trainer}"),
        seed=2
    )
    rlearn.set_config_to_trainer(conf, trainer)
    action_transformer = rlearn.transformer.DiscreteAction(actions=[0, 1])
    moving_r = 0
    env = gymnasium.make(
        'CartPole-v1',
        render_mode=render_mode
    )
    env.reset(seed=1)
    ep = 0
    for ep in range(conf.epochs):
        s, _ = env.reset()
        ep_r = 0
        value = None
        for _ in range(200):  # in one episode
            _a = trainer.predict(s)
            # IMPORTANT: it is better to record permuted action in buffer
            a = action_transformer.transform(_a)
            s_, _, done, _, _ = env.step(a)
            r = cartpole_reward(s_, env)
            # IMPORTANT: it is better to record permuted action in buffer
            trainer.store_transition(s, a, r, s_, done)
            s = s_
            ep_r += r
            if ep >= conf.not_learn_epochs:
                res = trainer.train_batch()
                value = res.value
            if done:
                trainer.trace({
                    "ep_reward": ep_r,
                }, step=ep)
                break
        moving_r = 0.05 * ep_r + 0.95 * moving_r
        print(f"{ep} r={ep_r:.2f}, mov={moving_r:.2f} value={value}")
        if moving_r > 30:
            break
    env.close()
    return ep


def train_pendulum(conf, render_mode="human", rnd=None):
    trainer = rlearn.get_trainer_by_name(
        conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), f"test_{conf.trainer}"),
        seed=3
    )
    if rnd is not None:
        trainer.add_rnd(target=rnd, learning_rate=1e-3)
    rlearn.set_config_to_trainer(conf, trainer)

    env = gymnasium.make(
        'Pendulum-v1',
        render_mode=render_mode
    )
    env.reset(seed=1)
    max_ep_step = 200
    moving = None
    ep = 0
    for ep in range(conf.epochs):
        s, _ = env.reset()
        ep_r = 0
        value = None
        for _ in range(max_ep_step):
            _a = trainer.predict(s)
            a = trainer.map_action(_a)
            s_, r, _, _, _ = env.step(a)
            # IMPORTANT: it is better to record permuted action in buffer
            trainer.store_transition(s, _a, (r + 8) / 8, s_)
            s = s_
            ep_r += r

            res = trainer.train_batch()
            value = res.value

        if ep % 20 == 0:
            dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
            trainer.save_model_weights(dir_)
            trainer.load_model_weights(dir_)
        trainer.trace({
            "ep_reward": ep_r,
        }, step=ep)
        if moving is None:
            moving = ep_r
        moving = moving * .95 + ep_r * .05
        print(f"{ep} r={ep_r:.2f} mov={moving:.2f} value={value}")
        if moving > -900:
            break
    env.close()
    return ep


def train_arcobot(conf, render_mode="human", rnd=None):
    trainer = rlearn.get_trainer_by_name(
        conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), f"test_{conf.trainer}"),
        seed=3
    )
    if rnd is not None:
        trainer.add_rnd(target=rnd, learning_rate=1e-3)
    rlearn.set_config_to_trainer(conf, trainer)

    env = gymnasium.make(
        'Acrobot-v1',
        render_mode=render_mode
    )
    env.reset(seed=1)
    max_ep_step = 200
    moving = None
    ep = 0
    for ep in range(conf.epochs):
        s, _ = env.reset()
        ep_r = 0
        value = None
        for _ in range(max_ep_step):  # in one episode
            _a = trainer.predict(s)
            # IMPORTANT: it is better to record permuted action in buffer
            a = trainer.map_action(_a)
            s_, r, _, _, _ = env.step(a)
            # IMPORTANT: it is better to record permuted action in buffer
            trainer.store_transition(s, _a, r, s_)
            s = s_
            ep_r += r

            res = trainer.train_batch()
            value = res.value

        if ep % 20 == 0:
            dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
            trainer.save_model_weights(dir_)
            trainer.load_model_weights(dir_)
        trainer.trace({
            "ep_reward": ep_r,
        }, step=ep)
        if moving is None:
            moving = ep_r
        moving = moving * .95 + ep_r * .05
        print(f"{ep} r={ep_r:.2f} mov={moving:.2f} value={value}")
        if moving > -100:
            break
    env.close()
    return ep


def train_mountain_car(conf, render_mode="human", rnd=None):
    trainer = rlearn.get_trainer_by_name(
        conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), f"test_{conf.trainer}"),
        seed=3
    )
    if rnd is not None:
        trainer.add_rnd(target=rnd, learning_rate=1e-4)
    rlearn.set_config_to_trainer(conf, trainer)
    action_transformer = rlearn.transformer.DiscreteAction([0, 1, 2])
    mov = 1000
    env = gymnasium.make(
        'MountainCar-v0',
        render_mode=render_mode,
    )
    env.reset(seed=1)
    ep = 0
    for ep in range(conf.epochs):
        s, _ = env.reset()
        step = 0
        while True:  # in one episode
            _a = trainer.predict(s)
            a = action_transformer.transform(_a)
            s_, _, done, _, _ = env.step(a)
            r = 1. if done else 0
            trainer.store_transition(s, a, r, s_, done)
            s = s_
            step += 1
            res = trainer.train_batch()
            if done:
                if ep % 20 == 0:
                    dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                    trainer.save_model_weights(dir_)
                    trainer.load_model_weights(dir_)
                break

        mov = mov * 0.8 + step * 0.2
        print(f"{ep} step={step} mov={mov}, value={res.value}")
        if mov < 700:
            break

    env.close()
    return ep


class GymTest(unittest.TestCase):
    def setUp(self) -> None:
        epochs = 200
        batch_size = 32
        buffer_size = 5000
        gamma = 0.9
        self.render_mode = "human"
        self.cartpole_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=batch_size,
            epochs=epochs,
            action_transform=[0, 1],
            nets=[
                rlearn.NetConfig(
                    input_shape=(4,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(4,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            gamma=gamma,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(buffer_size),
            replace_step=100,
            not_learn_epochs=2,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )
        self.acrobot_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=batch_size,
            epochs=epochs,
            action_transform=[0, 1, 2],
            nets=[
                rlearn.NetConfig(
                    input_shape=(6,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(6,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            gamma=gamma,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(buffer_size),
            replace_step=100,
            not_learn_epochs=2,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )
        self.pendulum_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=batch_size,
            epochs=epochs,
            action_transform=[[-2, 2]],
            replay_buffer=rlearn.ReplayBufferConfig(buffer_size),
            replace_step=200,
            nets=[
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            gamma=gamma,
            learning_rates=(0.01, 0.01),
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )
        self.mountain_car_conf = rlearn.TrainConfig(
            trainer="",
            batch_size=batch_size,
            epochs=epochs,
            action_transform=[0, 1, 2],
            nets=[
                rlearn.NetConfig(
                    input_shape=(2,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(2,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 32}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            gamma=gamma,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(buffer_size),
            replace_step=100,
            not_learn_epochs=2,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), "tmp"), ignore_errors=True)

    def test_dqn(self):
        # 29 r=123.33, mov=31.88 value={'loss': 0.008630372, 'q': 4.0718727}
        self.cartpole_conf.trainer = rlearn.DQNTrainer.name
        ep = train_cartpole(self.cartpole_conf, self.render_mode)
        self.assertLess(ep, 50)

    def test_dqn_acrobot(self):
        # 29 r=123.33, mov=31.88 value={'loss': 0.008630372, 'q': 4.0718727}
        self.acrobot_conf.trainer = rlearn.DQNTrainer.name
        rnd = keras.Sequential([
            keras.layers.InputLayer(6),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(8),
        ])
        ep = train_arcobot(self.acrobot_conf, self.render_mode, rnd=rnd)
        self.assertLess(ep, 100)

    def test_dueling_dqn(self):
        # 18 r=97.65, mov=30.95 value={'loss': 0.03248573, 'q': 3.6843674}
        self.cartpole_conf.trainer = rlearn.DuelingDQNTrainer.name
        ep = train_cartpole(self.cartpole_conf, self.render_mode)
        self.assertLess(ep, 50)

    def test_ppo_discrete(self):
        # 38 r=98.69, mov=32.19 value={'pi_loss': -0.014326467, 'critic_loss': 0.10958307, 'reward': 4.6177177}
        self.cartpole_conf.trainer = rlearn.PPODiscreteTrainer.name
        self.cartpole_conf.learning_rates = (0.01, 0.01)
        ep = train_cartpole(self.cartpole_conf, self.render_mode)
        self.assertLess(ep, 50)

    def test_ddpg(self):
        # 21 r=-261.60 mov=-888.00 value={'actor_loss': -5.081643, 'critic_loss': 0.017063033}
        self.pendulum_conf.trainer = rlearn.DDPGTrainer.name
        ep = train_pendulum(self.pendulum_conf, self.render_mode)
        self.assertLess(ep, 50)

    def test_td3(self):
        # 42 r=-401.56 mov=-888.84 value={'actor_loss': -6.233742, 'critic_loss': 0.028469825, 'reward': 0.5677}
        self.pendulum_conf.trainer = rlearn.TD3Trainer.name
        ep = train_pendulum(self.pendulum_conf, self.render_mode)
        self.assertLess(ep, 50)

    def test_ppo_continuous(self):
        # 46 r=-517.77 mov=-888.73 value={'pi_loss': 0.13905787, 'critic_loss': 0.017424757}
        self.pendulum_conf.trainer = rlearn.PPOContinueTrainer.name
        ep = train_pendulum(self.pendulum_conf, self.render_mode)
        self.assertLess(ep, 60)

    def test_prioritized_dqn(self):
        # 10 step=174 mov=629.4898035712, value={'loss': 0.0005622931, 'q': 0.82957065}
        self.mountain_car_conf.trainer = rlearn.DQNTrainer.name
        self.mountain_car_conf.replay_buffer = rlearn.ReplayBufferConfig(10000, buf="PrioritizedReplayBuffer")
        ep = train_mountain_car(self.mountain_car_conf, self.render_mode)
        self.assertLess(ep, 20)

    def test_rnd_dqn(self):
        # 10 step=174 mov=629.4898035712, value={'loss': 0.0005622931, 'q': 0.82957065}
        self.mountain_car_conf.trainer = rlearn.DQNTrainer.name
        rnd = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(8),
        ])
        ep = train_mountain_car(self.mountain_car_conf, self.render_mode, rnd=rnd, )
        self.assertLess(ep, 20)

    def test_actor_critic(self):
        self.cartpole_conf.trainer = rlearn.ActorCriticDiscreteTrainer.name
        self.cartpole_conf.learning_rates = (0.01, 0.01)
        ep = train_cartpole(self.cartpole_conf, self.render_mode)
        self.assertLess(ep, 90)

    def test_sac_discrete(self):
        self.cartpole_conf.trainer = rlearn.SACDiscreteTrainer.name
        self.cartpole_conf.learning_rates = (0.01, 0.01)
        ep = train_cartpole(self.cartpole_conf, self.render_mode)
        self.assertLess(ep, 90)

    def test_sac_continuous(self):
        self.pendulum_conf.trainer = rlearn.SACContinueTrainer.name
        ep = train_pendulum(self.pendulum_conf, self.render_mode)
        self.assertLess(ep, 60)

    def test_sac_rnd_discrete(self):
        rnd = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(8),
        ])
        self.mountain_car_conf.trainer = rlearn.SACDiscreteTrainer.name
        self.mountain_car_conf.learning_rates = (0.01, 0.01)
        ep = train_mountain_car(self.mountain_car_conf, self.render_mode, rnd=rnd)
        self.assertLess(ep, 20)

    def test_ppo_rnd_discrete(self):
        rnd = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(8),
        ])
        self.mountain_car_conf.trainer = rlearn.PPODiscreteTrainer.name
        self.mountain_car_conf.learning_rates = (0.01, 0.01)
        ep = train_mountain_car(self.mountain_car_conf, self.render_mode, rnd=rnd)
        self.assertLess(ep, 20)


class ExperienceDistributedGym(unittest.TestCase):
    def setUp(self) -> None:
        self.result_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "dist_gym_test")
        self.ps = []
        buf_port = tools.get_available_port()
        self.buf_address = f'127.0.0.1:{buf_port}'
        p = multiprocessing.Process(target=rlearn.distributed.experience.start_replay_buffer_server, kwargs=dict(
            port=buf_port,
            max_size=10000,
            buf="RandomReplayBuffer",
            # debug=True,
        ))
        p.start()
        self.ps.append(p)

    def tearDown(self) -> None:
        [p.join() for p in self.ps]
        [p.terminate() for p in self.ps]
        shutil.rmtree(self.result_dir, ignore_errors=True)

    def set_actors(self, env: rlearn.EnvWrapper, n_actors=2, local_buffer_size=50):
        actors_address = []
        for _ in range(n_actors):
            actor_port = tools.get_available_port()
            p = multiprocessing.Process(target=rlearn.distributed.experience.start_actor_server, kwargs=dict(
                port=actor_port,
                remote_buffer_address=self.buf_address,
                local_buffer_size=local_buffer_size,
                env=env,
                # debug=True,
            ))
            p.start()
            self.ps.append(p)
            actor_address = f'127.0.0.1:{actor_port}'
            actors_address.append(actor_address)
        return actors_address

    def test_dqn(self):
        env = gym_wrapper_test.CartPoleDiscreteReward(render_mode="human")
        actors_address = self.set_actors(env, n_actors=2)

        trainer = rlearn.trainer.DQNTrainer()
        trainer.set_replay_buffer(max_size=5000)
        trainer.set_model_encoder(
            q=keras.Sequential([
                keras.layers.InputLayer(4),
                keras.layers.Dense(20),
                keras.layers.ReLU(),
            ]),
            action_num=2
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=100,
        )
        learner = rlearn.distributed.experience.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=30, epoch_step=None, replicate_step=100)

    def test_ddpg(self):
        env = gym_wrapper_test.Pendulum(render_mode="human")
        actors_address = self.set_actors(env, n_actors=2)

        trainer = rlearn.trainer.DDPGTrainer()
        trainer.set_replay_buffer(max_size=3000)
        trainer.set_action_transformer(rlearn.transformer.ContinuousAction([-2, 2]))
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=1,
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=100,
        )
        learner = rlearn.distributed.experience.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=50, epoch_step=None, replicate_step=100)

    def test_ppo_discrete(self):
        env = gym_wrapper_test.CartPoleDiscreteReward(render_mode="human")
        actors_address = self.set_actors(env, n_actors=2, local_buffer_size=100)

        trainer = rlearn.trainer.PPODiscreteTrainer()
        trainer.set_replay_buffer(max_size=3000)
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer(4),
                keras.layers.Dense(20),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(4),
                keras.layers.Dense(20),
                keras.layers.ReLU(),
            ]),
            action_num=2,
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=100,
        )
        learner = rlearn.distributed.experience.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=200, epoch_step=None)

    def test_ppo_continue(self):
        env = gym_wrapper_test.Pendulum(render_mode="human")
        actors_address = self.set_actors(
            env, n_actors=2, local_buffer_size=100)

        trainer = rlearn.trainer.PPOContinueTrainer()
        trainer.set_replay_buffer(max_size=2000)
        trainer.set_action_transformer(rlearn.transformer.ContinuousAction([-2, 2]))
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(32),
                keras.layers.ReLU(),
            ]),
            action_num=1,
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=100,
        )
        learner = rlearn.distributed.experience.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=200, epoch_step=None)
