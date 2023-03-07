import shutil

import rlearn.supervised
from rl_maze import *


def short_gem(graph, gems, me):
    short_path = None
    short_count = 9999999
    for v in gems.values():
        g = v[0]
        path = pathfind.find(graph=graph, start=f'{me.row},{me.col}', end=f"{g.row},{g.col}")
        if short_path is None:
            short_path = path
        new_count = len(path)
        if short_count > new_count:
            short_count = new_count
            short_path = path
    return short_path


def get_data():
    env = Maze()
    a_map = {s: i for i, s in enumerate(ACTION_ORDER)}
    x_data, y_data = [], []
    for ep in range(30):
        raw_s = env.reset()
        maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        me = raw_s["players"][raw_s["my_id"]]
        exit = raw_s["exits"][raw_s["my_id"]]
        players = raw_s["players"]
        gems = raw_s["gems"]
        s = parse_state3(
            me=me,
            players=players,
            gems=gems,
            maze=maze
        )

        while True:
            # env.render()

            e_path = pathfind.find(graph=gp, start=f'{me.row},{me.col}', end=f"{exit.row},{exit.col}")
            g_path = short_gem(graph=gp, gems=raw_s["gems"], me=me)
            if len(e_path) - 1 > me.energy:
                next_n = e_path[1].split(",")
            else:
                next_n = g_path[1].split(",")
            nr, nc = int(next_n[0]), int(next_n[1])
            action = move(nr, nc, me.row, me.col)
            a_int = a_map[action]
            if len(e_path) < me.energy:
                x_data.append(s[None, :])
                y_data.append(a_int)
            raw_s, _, done = env.step(action)
            me = raw_s["players"][raw_s["my_id"]]
            exit = raw_s["exits"][raw_s["my_id"]]
            players = raw_s["players"]
            gems = raw_s["gems"]
            s = parse_state3(
                me=me,
                players=players,
                gems=gems,
                maze=maze)
            if done:
                break
        print(f"collect data {ep=}")
    x = np.concatenate(x_data, axis=0, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(os.path.join(TMP_DIR, "mazeData.npz"), "wb") as f:
        np.savez(f, x=x, y=y)
    env.close()
    return x, y


def train_super(ep):
    trainer = build_trainer()
    data = np.load(os.path.join(TMP_DIR, "mazeData.npz"))
    rlearn.supervised.fit(
        trainer=trainer, x=data["x"], y=data["y"], epoch=ep, learning_rate=1e-3,
        model_save_dir=os.path.join(TMP_DIR, "superModel"), verbose=1)
    return trainer


def predict(load_ep):
    trainer = build_trainer()
    rlearn.supervised.set_actor_weights(trainer, os.path.join(TMP_DIR, "superModel", f"ep-{load_ep}.zip"))
    env = Maze()
    for ep in range(2):
        raw_s = env.reset()
        maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        me = raw_s["players"][raw_s["my_id"]]
        exit = raw_s["exits"][raw_s["my_id"]]
        players = raw_s["players"]
        gems = raw_s["gems"]
        s = parse_state3(
            me=me,
            players=players,
            gems=gems,
            maze=maze
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
            me = raw_s["players"][raw_s["my_id"]]
            exit = raw_s["exits"][raw_s["my_id"]]
            players = raw_s["players"]
            gems = raw_s["gems"]
            ep_r += r
            s = parse_state3(
                me=me,
                players=players,
                gems=gems,
                maze=maze)
            if done:
                break
        print(f"predict ep={ep}, ep_r={ep_r}")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--get_data", action="store_true")
    parser.add_argument("-d", "--display", action="store_true")
    args = parser.parse_args()

    if not args.display:
        os.environ['SDL_VIDEODRIVER'] = "dummy"
    if args.get_data:
        get_data()

    super_train_ep = 20
    shutil.rmtree(os.path.join(TMP_DIR, "superModel"), ignore_errors=True)
    train_super(super_train_ep)
    if args.display:
        predict(load_ep=super_train_ep)
