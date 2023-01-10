import os
import shutil
import tempfile
import unittest

import gym

import rlearn


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
            conf.trainer, log_dir=os.path.join(tempfile.tempdir, "test_dqn"),
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
            conf.trainer, log_dir=os.path.join(tempfile.tempdir, "test_dueling_dqn"),
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
            epochs=10,
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
        trainer: rlearn.PPOContinueTrainer = rlearn.get_trainer_by_name(
            conf.trainer, log_dir=os.path.join(tempfile.tempdir, "test_ppo"),
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
                s_, r, _, _, _ = env.step(real_a[0])
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
            conf.trainer, log_dir=os.path.join(tempfile.tempdir, "test_p_dqn"),
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
            if sum(ep_pass_record[-6:]) == 4:
                passed = True
                break

        self.assertTrue(passed, msg=f"{ep_pass_record=}")
        env.close()
