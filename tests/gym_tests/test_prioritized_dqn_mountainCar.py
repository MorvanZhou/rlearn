import os

import gym

import rlearn

conf = rlearn.TrainConfig(
    trainer=rlearn.DQNTrainer.name,
    batch_size=16,
    epochs=1000,
    action_transform=[0, 1, 2],
    nets=[rlearn.NetConfig(
        input_shape=(2,),
        layers=[
            rlearn.LayerConfig("dense", args={"units": 32}),
            rlearn.LayerConfig("relu"),
        ]
    )],
    gamma=0.9,
    learning_rates=(0.01,),
    replay_buffer=rlearn.ReplayBufferConfig(10000, rlearn.PrioritizedReplayBuffer.name),
    replace_step=500,
    not_learn_epochs=0,
    epsilon_decay=0.001,
    min_epsilon=0.1,
)

trainer = rlearn.get_trainer_by_name(
    conf.trainer, conf.learning_rates,
    log_dir=os.path.join(os.pardir, os.pardir, "tmp", "test_prioritized_dqn"),
    seed=1
)
rlearn.set_config_to_trainer(conf, trainer)

env = gym.make('MountainCar-v0', new_step_api=True)
env.reset(seed=1)
for ep in range(conf.epochs):
    s = env.reset(return_info=False)
    ep_r = 0.
    loss = 0
    q_max = 0
    step_count = 0
    while True:  # in one episode
        a = trainer.predict(s)
        s_, _, done, _, _ = env.step(a)
        r = 1. if done else 0.
        trainer.store_transition(s, a, r, s_)
        s = s_
        ep_r += r
        if ep >= conf.not_learn_epochs:
            res = trainer.train_batch()
            l_, q_max = res["loss"], res["q"]
            if l_ > loss:
                loss = l_
        step_count += 1
        if done:
            if ep % 20 == 0:
                dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                trainer.save_model(dir_)
                trainer.load_model_weights(dir_)
            trainer.trace({
                "ep_reward": ep_r,
                "loss": loss,
                "q_max": q_max,
                "epsilon": trainer.epsilon
            }, step=ep)
            break

    print(f"{ep} step_count={step_count}"
          f" r={ep_r:.2f}, max_loss={loss:.4f} qmax={q_max:.4f} epsilon={trainer.epsilon:.3f}")

env.close()
