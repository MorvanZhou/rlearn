import os

import gym

import rlearn

conf = rlearn.TrainConfig(
    trainer=rlearn.PPODiscreteTrainer.name,
    batch_size=32,
    epochs=10000,
    action_transform=[0, 1],
    replay_buffer=rlearn.ReplayBufferConfig(1000),
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
    gamma=0.9,
    learning_rates=(0.001, 0.01),
    replace_step=0,
)

trainer = rlearn.PPODiscreteTrainer(
    learning_rates=conf.learning_rates,
    log_dir=os.path.join(os.pardir, os.pardir, "tmp", "test_ppo_discrete"))
rlearn.set_config_to_trainer(conf, trainer)

env = gym.make('CartPole-v1', new_step_api=True, render_mode="human")
max_ep_step = 200
for ep in range(conf.epochs):
    s = env.reset(return_info=False)
    ep_r = 0
    q_max = 0
    a_loss, c_loss = 0, 0
    for t in range(max_ep_step):  # in one episode
        a = trainer.predict(s)
        s_, _, done, _, _ = env.step(int(a))
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        trainer.store_transition(s, a, r, done)
        s = s_
        ep_r += r

        # update ppo
        learn_res = trainer.train_batch()
        a_loss = learn_res["a_loss"]
        c_loss = learn_res["c_loss"]
        if done:
            trainer.trace({
                "ep_reward": ep_r,
                "a_loss": a_loss,
                "c_loss": c_loss,
                "epsilon": trainer.epsilon
            }, step=ep)
        if trainer.train_step % 20 == 0:
            trainer.update_policy()
        if done:
            break

    if ep % 20 == 0:
        dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
        trainer.save_model(dir_)
        trainer.load_model_weights(dir_)

    print(f"{ep} r={ep_r:.2f}, a={a_loss:.4f} c={c_loss:.4f}")

env.close()
