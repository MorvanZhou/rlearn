
def on_cart_pole_v1(update, conf, env, map_data):
    for ep in range(conf.epochs):
        s = env.reset(return_info=False)
        done = False
        for t in range(map_data["steps"]):
            ctx = {
                "s": s,
                "env": env,
                "step": t,
                "epoch": ep,
                "done": done
            }

            a = update(ctx)
            if done:
                break

            s, _, done, _, _ = env.step(a)


def off_cart_pole_v1(update, conf, env, map_data):
    for ep in range(conf.epochs):
        s = env.reset(return_info=False)
        done = False
        for t in range(map_data["steps"]):
            ctx = {
                "s": s,
                "env": env,
                "step": t,
                "epoch": ep,
                "done": done
            }

            a = update(ctx)
            if done:
                break

            s, _, done, _, _ = env.step(a)


def on_pendulum_v1(update, conf, env, map_data):
    for ep in range(conf.epochs):
        s = env.reset(return_info=False)
        r = 0
        for t in range(map_data["steps"]):
            done = t == map_data["steps"] - 1
            ctx = {
                "s": s,
                "env": env,
                "step": t,
                "epoch": ep,
                "r": r,
                "done": done
            }

            a = update(ctx)

            s, r, _, _, _ = env.step(a)


def off_pendulum_v1(update, conf, env, map_data):
    for ep in range(conf.epochs):
        s = env.reset(return_info=False)
        r = 0
        for t in range(map_data["steps"]):
            done = t == map_data["steps"] - 1
            ctx = {
                "s": s,
                "env": env,
                "step": t,
                "epoch": ep,
                "r": r,
                "done": done
            }

            a = update(ctx)

            s, r, _, _, _ = env.step(a)


ON_POLICY_ENV_JOB_MAP = {
    "CartPole-v1": on_cart_pole_v1,
    "Pendulum-v1": on_pendulum_v1,
}

OFF_POLICY_ENV_JOB_MAP = {
    "CartPole-v1": off_cart_pole_v1,
    "Pendulum-v1": off_pendulum_v1,
}
