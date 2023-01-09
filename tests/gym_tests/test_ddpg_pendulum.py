import os

import gym

import rlearn

conf = rlearn.TrainConfig(
    trainer=rlearn.DDPGTrainer.name,
    batch_size=32,
    epochs=1000,
    action_transform=[[-2, 2]],
    nets=[
        rlearn.NetConfig(
            input_shape=(3,),
            layers=[
                rlearn.LayerConfig("dense", args={"units": 30, }),
                rlearn.LayerConfig("relu"),
            ]),
        rlearn.NetConfig(
            input_shape=(3,),
            layers=[
                rlearn.LayerConfig("dense", args={"units": 30}),
                rlearn.LayerConfig("relu"),
            ]
        )
    ],
    gamma=0.9,
    learning_rates=(0.001, 0.01),
    replay_buffer=rlearn.ReplayBufferConfig(3000),
    not_learn_epochs=0,
    epsilon_decay=0.01,
    min_epsilon=0.1,
    replace_step=200,
    replace_ratio=1.,  # tau
)

trainer = rlearn.DDPGTrainer(
    learning_rates=conf.learning_rates,
    log_dir=os.path.join(os.pardir, os.pardir, "tmp", "test_ddpg")
)
rlearn.set_config_to_trainer(conf, trainer)

action_transformer = rlearn.transformer.ContinuousAction(conf.action_transform)

env = gym.make('Pendulum-v1', new_step_api=True, render_mode="human")
MAX_EP_STEP = 200

for ep in range(200):
    s = env.reset(seed=1)  # 设置随机种子
    ep_reward = 0
    for step in range(MAX_EP_STEP):
        raw_a = trainer.predict(s)
        real_a = action_transformer.transform(raw_a)
        s_, r, _, _, _ = env.step(real_a)

        # 将当前的状态,行为,回报,下一个状态存储到记忆库中
        trainer.store_transition(s, raw_a, r / 10, s_)

        s = s_
        ep_reward += r

        if ep >= conf.not_learn_epochs:
            res = trainer.train_batch()
            a_loss, c_loss = res["a_loss"], res["c_loss"]
            if step == MAX_EP_STEP - 1:
                trainer.trace({
                    "ep_reward": ep_reward,
                    "a_loss": a_loss,
                    "c_loss": c_loss,
                    "epsilon": trainer.epsilon
                }, step=ep)
                print(f"{ep} {ep_reward=:.2f}, {a_loss=:.4f}, {c_loss=:.4f}")
                if ep % 20 == 0:
                    dir_ = os.path.join(trainer.log_dir, "checkpoints", f"ep-{ep:06d}")
                    trainer.save_model_weights(dir_)
                    trainer.load_model_weights(dir_)
                break

env.close()  # 关闭渲染窗口
