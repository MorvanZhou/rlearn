from rl_maze import *


def deploy_rl(load_ep):
    env = Maze()
    trainer = build_trainer()
    trainer.load_model_weights(os.path.join(TMP_DIR, "rlModel", f"ep-{load_ep}.zip"))

    for ep in range(300):
        raw_s = env.reset()
        # maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        me, players, exit, items = raw_state_convert(raw_s)
        s = parse_state4(
            me=me,
            players=players,
            items=items,
            graph=gp
        )

        ep_r = 0
        while True:
            env.render()
            path = pathfind.find(graph=gp, start=f'{me.row},{me.col}', end=f"{exit.row},{exit.col}")
            if len(path) - 1 <= me.energy:
                action = trainer.model.mapped_predict(s)
                dr, dc = ACTION_MOVE_DELTA[action]
                if me.row + dr == exit.row and me.col + dc == exit.col:
                    action = "s"
            else:
                if len(path) == 0:
                    action = "s"
                else:
                    next_n = path[1].split(",")
                    nr, nc = int(next_n[0]), int(next_n[1])
                    action = move(nr, nc, me.row, me.col)
            raw_s, r, done = env.step(action)
            me, players, exit, items = raw_state_convert(raw_s)
            ep_r += r
            s = parse_state4(
                me=me,
                players=players,
                items=items,
                graph=gp
            )
            if done:
                break

        print(f"{ep=}, {ep_r=:.3f}, {trainer.epsilon=:.3f}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", type=int)
    args = parser.parse_args()
    deploy_rl(load_ep=args.load)
