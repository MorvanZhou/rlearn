import multiprocessing
import os
import shutil
import tempfile
import unittest

import gym
from tensorflow import keras

import rlearn
from rlearn.distribute import tools


class GymTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), "tmp"), ignore_errors=True)

    def test_dqn(self):
        conf = rlearn.TrainConfig(
            trainer=rlearn.DQNTrainer.name,
            batch_size=32,
            epochs=10,
            action_transform=[0, 1],
            nets=[rlearn.NetConfig(
                input_shape=(4,),
                layers=[
                    rlearn.LayerConfig("dense", args={"units": 20}),
                    rlearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.9,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(500),
            replace_step=100,
            not_learn_epochs=2,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={}
        )
        trainer = rlearn.get_trainer_by_name(
            conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), "test_dqn"),
            seed=2
        )
        rlearn.set_config_to_trainer(conf, trainer)

        moving_r = 0
        env = gym.make('CartPole-v1', new_step_api=True)
        env.reset(seed=1)
        for ep in range(conf.epochs):
            s = env.reset(return_info=False)
            ep_r = 0
            loss = 0
            q_max = 0
            for _ in range(200):  # in one episode
                a = trainer.predict(s)
                s_, _, done, _, _ = env.step(a)
                x, _, theta, _ = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
                trainer.store_transition(s, a, r, s_)
                s = s_
                ep_r += r
                if ep >= conf.not_learn_epochs:
                    res = trainer.train_batch()
                    loss, q_max = res["loss"], res["q"]
                if done:
                    trainer.trace({
                        "ep_reward": ep_r,
                        "loss": loss,
                        "q_max": q_max,
                        "epsilon": trainer.epsilon
                    }, step=ep)
                    break
            moving_r = 0.2 * ep_r + 0.8 * moving_r
            print(f"{ep} r={ep_r:.2f}, loss={loss:.4f} qmax={q_max:.4f}")
        self.assertGreater(moving_r, 10, msg=f"{moving_r=}")
        env.close()

    def test_dueling_dqn(self):
        conf = rlearn.TrainConfig(
            trainer=rlearn.DuelingDQNTrainer.name,
            batch_size=16,
            epochs=10,
            action_transform=[0, 1],
            nets=[rlearn.NetConfig(
                input_shape=(4,),
                layers=[
                    rlearn.LayerConfig("dense", args={"units": 20}),
                    rlearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.9,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(500),
            replace_step=100,
            not_learn_epochs=2,
            epsilon_decay=0.05,
            min_epsilon=0.1,
            args={}
        )
        trainer = rlearn.get_trainer_by_name(
            conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), "test_dueling_dqn"),
            seed=4
        )
        rlearn.set_config_to_trainer(conf, trainer)

        env = gym.make('CartPole-v1', new_step_api=True)
        env.reset(seed=1)
        moving_r = 0
        for ep in range(conf.epochs):
            s = env.reset(return_info=False)
            ep_r = 0
            loss = 0
            q_max = 0
            for _ in range(200):  # in one episode
                a = trainer.predict(s)
                s_, _, done, _, _ = env.step(a)
                x, _, theta, _ = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
                trainer.store_transition(s, a, r, s_)
                s = s_
                ep_r += r
                if ep >= conf.not_learn_epochs:
                    res = trainer.train_batch()
                    loss, q_max = res["loss"], res["q"]
                if done:
                    if ep % 20 == 0:
                        dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                        trainer.save_model_weights(dir_)
                        trainer.load_model_weights(dir_)
                    trainer.trace({
                        "ep_reward": ep_r,
                        "loss": loss,
                        "q_max": q_max,
                        "epsilon": trainer.epsilon
                    }, step=ep)
                    break
            moving_r = 0.2 * ep_r + 0.8 * moving_r
            print(f"{ep} r={ep_r:.2f}, loss={loss:.4f} qmax={q_max:.4f}")
        self.assertGreater(moving_r, 6, msg=f"{moving_r=}")
        env.close()

    def test_ppo(self):
        conf = rlearn.TrainConfig(
            trainer=rlearn.PPOContinueTrainer.name,
            batch_size=16,
            epochs=3,
            action_transform=[[-2, 2]],
            replay_buffer=rlearn.ReplayBufferConfig(1000),
            replace_step=100,
            nets=[
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 20}),
                        rlearn.LayerConfig("relu"),
                    ]
                ),
                rlearn.NetConfig(
                    input_shape=(3,),
                    layers=[
                        rlearn.LayerConfig("dense", args={"units": 20}),
                        rlearn.LayerConfig("relu"),
                    ]
                )
            ],
            gamma=0.9,
            learning_rates=(0.01, 0.01),
            args={}
        )
        trainer = rlearn.get_trainer_by_name(
            conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), "test_ppo"),
            seed=4
        )
        rlearn.set_config_to_trainer(conf, trainer)
        action_transformer = rlearn.transformer.ContinuousAction(conf.action_transform)

        env = gym.make('Pendulum-v1', new_step_api=True)
        env.reset(seed=1)
        max_ep_step = 200
        learn_step = 0
        for ep in range(conf.epochs):
            s = env.reset(return_info=False)
            ep_r = 0
            a_loss, c_loss = 0, 0
            for t in range(max_ep_step):  # in one episode
                raw_a = trainer.predict(s)
                real_a = action_transformer.transform(raw_a)
                s_, r, _, _, _ = env.step(real_a)
                done = t == max_ep_step - 1
                trainer.store_transition(s, raw_a, (r + 8) / 8, done)
                s = s_
                ep_r += r

                # update ppo
                res = trainer.train_batch()
                learn_step += 1
                a_loss, c_loss = res["a_loss"], res["c_loss"]

            if ep % 20 == 0:
                dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                trainer.save_model_weights(dir_)
                trainer.load_model_weights(dir_)
            trainer.trace({
                "ep_reward": ep_r,
                "loss_actor": a_loss,
                "loss_critic": c_loss,
            }, step=ep)
            print(f"{ep} r={ep_r:.2f}, a={a_loss:.4f} c={c_loss:.4f}")

        env.close()

    def test_prioritized_dqn(self):
        conf = rlearn.TrainConfig(
            trainer=rlearn.DQNTrainer.name,
            batch_size=16,
            epochs=10,
            action_transform=[0, 1, 2],
            nets=[rlearn.NetConfig(
                input_shape=(2,),
                layers=[
                    rlearn.LayerConfig("dense", args={"units": 20}),
                    rlearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.95,
            learning_rates=(0.001,),
            replay_buffer=rlearn.ReplayBufferConfig(10000),
            replace_step=600,
            not_learn_epochs=0,
            epsilon_decay=0.1,
            min_epsilon=0.1,
            args={"buffer": rlearn.PrioritizedReplayBuffer.name, "seed": 8}
        )

        trainer = rlearn.get_trainer_by_name(
            conf.trainer, log_dir=os.path.join(tempfile.gettempdir(), "test_p_dqn"),
            seed=4
        )
        rlearn.set_config_to_trainer(conf, trainer)
        passed = False
        ep_pass_record = []
        env = gym.make('MountainCar-v0', new_step_api=True)
        env.reset(seed=1)
        for ep in range(conf.epochs):
            s = env.reset(return_info=False)
            ep_r = 0.
            loss = 0
            q_max = 0
            for t in range(200):  # in one episode
                a = trainer.predict(s)
                s_, _, done, _, _ = env.step(a)
                r = 10 if done else 1 if s_[0] > 0.33 else 0
                trainer.store_transition(s, a, r, s_)
                s = s_
                ep_r += r
                if ep >= conf.not_learn_epochs:
                    res = trainer.train_batch()
                    loss, q_max = res["loss"], res["q"]
                if done:
                    if ep % 20 == 0:
                        dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                        trainer.save_model_weights(dir_)
                        trainer.load_model_weights(dir_)
                    trainer.trace({
                        "ep_reward": ep_r,
                        "loss": loss,
                        "q_max": q_max,
                        "epsilon": trainer.epsilon
                    }, step=ep)
                    break

            print(f"{ep} r={ep_r:.2f}, loss={loss:.4f} qmax={q_max:.4f}")
            if t < 199:
                ep_pass_record.append(1)
            else:
                ep_pass_record.append(0)
            if sum(ep_pass_record[-6:]) >= 1:
                passed = True
                break

        self.assertTrue(passed, msg=f"{ep_pass_record=}")
        env.close()


class DistributedGym(unittest.TestCase):
    def setUp(self) -> None:
        self.result_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "dist_gym_test")
        self.ps = []
        buf_port = tools.get_available_port()
        self.buf_address = f'127.0.0.1:{buf_port}'
        p = multiprocessing.Process(target=rlearn.distribute.start_replay_buffer_server, kwargs=dict(
            port=buf_port,
            max_size=10000,
            buf="RandomReplayBuffer",
            debug=True,
        ))
        p.start()
        self.ps.append(p)

    def tearDown(self) -> None:
        [p.terminate() for p in self.ps]
        shutil.rmtree(self.result_dir, ignore_errors=True)

    def set_actors(self, env: rlearn.EnvWrapper, n_actors=2, local_buffer_size=50, action_transformer=None):
        actors_address = []
        for _ in range(n_actors):
            actor_port = tools.get_available_port()
            p = multiprocessing.Process(target=rlearn.distribute.start_actor_server, kwargs=dict(
                port=actor_port,
                remote_buffer_address=self.buf_address,
                local_buffer_size=local_buffer_size,
                env=env,
                action_transformer=action_transformer,
                # debug=True,
            ))
            p.start()
            self.ps.append(p)
            actor_address = f'127.0.0.1:{actor_port}'
            actors_address.append(actor_address)
        return actors_address

    def test_dqn(self):
        env = gym_wrapper.CartPoleDiscreteReward(render_mode="human")
        actors_address = self.set_actors(env)

        trainer = rlearn.trainer.DQNTrainer()
        trainer.set_replay_buffer(max_size=1000)
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
        learner = rlearn.distribute.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=100, epoch_step=None, replicate_step=100)

    def test_ddpg(self):
        env = gym_wrapper.Pendulum(render_mode="human")
        action_transformer = rlearn.transformer.ContinuousAction([[-2, 2]])
        actors_address = self.set_actors(env, n_actors=2, action_transformer=action_transformer)

        trainer = rlearn.trainer.DDPGTrainer()
        trainer.set_replay_buffer(max_size=2000)
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
        learner = rlearn.distribute.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=300, epoch_step=None, replicate_step=100)

    def test_ppo_discrete(self):
        env = gym_wrapper.CartPoleDiscreteReward(render_mode="human")
        actors_address = self.set_actors(env, n_actors=5, local_buffer_size=200)

        trainer = rlearn.trainer.PPODiscreteTrainer()
        trainer.set_replay_buffer(max_size=2000)
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
        learner = rlearn.distribute.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=500, epoch_step=None)

    def test_ppo_continue(self):
        env = gym_wrapper.Pendulum(render_mode="human")
        action_transformer = rlearn.transformer.ContinuousAction([[-2, 2]])
        actors_address = self.set_actors(env, n_actors=5, local_buffer_size=200, action_transformer=action_transformer)

        trainer = rlearn.trainer.PPOContinueTrainer()
        trainer.set_replay_buffer(max_size=2000)
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(20),
                keras.layers.ReLU(),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(3),
                keras.layers.Dense(20),
                keras.layers.ReLU(),
            ]),
            action_num=1,
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=100,
        )
        learner = rlearn.distribute.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=500, epoch_step=None)
