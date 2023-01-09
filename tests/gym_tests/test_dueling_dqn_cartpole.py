import os

import gym

import rlearn

conf = rlearn.TrainConfig(
    trainer=rlearn.DuelingDQNTrainer.name,
    batch_size=32,
    epochs=1000,
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
    not_learn_epochs=5,
    epsilon_decay=0.01,
    min_epsilon=0.1,
)

trainer = rlearn.DuelingDQNTrainer(
    conf.learning_rates,
    log_dir=os.path.join(os.pardir, os.pardir, "tmp", "test_dueling_dqn")
)
rlearn.set_config_to_trainer(conf, trainer)

env = gym.make('CartPole-v1', new_step_api=True, render_mode="human")
for ep in range(conf.epochs):
    s = env.reset(return_info=False)
    ep_r = 0
    loss = 0
    q_max = 0
    for t in range(200):  # in one episode
        a = trainer.predict(s)
        s_, _, done, _, _ = env.step(a)
        x, x_dot, theta, theta_dot = s_
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

    print(f"{ep} r={ep_r:.2f}, loss={loss:.4f} qmax={q_max:.4f}")

env.close()
