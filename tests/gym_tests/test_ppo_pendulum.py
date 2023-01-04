import os

import gym

import rllearn

conf = rllearn.TrainConfig(
    trainer=rllearn.PPOContinueTrainer.name,
    batch_size=32,
    epochs=1000,
    action_transform=[[-2, 2]],
    replay_buffer=rllearn.ReplayBufferConfig(2000),
    nets=[
        rllearn.NetConfig(
            input_shape=(3,),
            layers=[
                rllearn.LayerConfig("dense", args={"units": 32}),
                rllearn.LayerConfig("relu"),
            ]
        ),
        rllearn.NetConfig(
            input_shape=(3,),
            layers=[
                rllearn.LayerConfig("dense", args={"units": 32}),
                rllearn.LayerConfig("relu"),
            ]
        )
    ],
    gamma=0.9,
    learning_rates=(0.001, 0.01),
    replace_step=200,
)

trainer = rllearn.PPOContinueTrainer(
    learning_rates=conf.learning_rates,
    log_dir=os.path.join(os.pardir, os.pardir, "tmp", "test_ppo_continuous"))
rllearn.set_config_to_trainer(conf, trainer)

action_transformer = rllearn.transformer.ContinuousAction(conf.action_transform)

env = gym.make('Pendulum-v1', new_step_api=True, render_mode="human")
max_ep_step = 200
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
        a_loss, c_loss = res["a_loss"], res["c_loss"]

    if ep % 20 == 0:
        dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
        trainer.save_model(dir_)
        trainer.load_model_weights(dir_)
    trainer.trace({
        "ep_reward": ep_r,
        "loss_actor": a_loss,
        "loss_critic": c_loss,
    }, step=ep)
    print(f"{ep} r={ep_r:.2f}, a={a_loss:.4f} c={c_loss:.4f}")

env.close()
