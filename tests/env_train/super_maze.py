from .rl_maze import *


def short_gem(graph, gems, me):
    short_path = None
    short_count = 9999999
    for v in gems.values():
        grow = v["x"]
        gcol = v["y"]
        mp = me["position"]
        mrow = mp["x"]
        mcol = mp["y"]
        path = pathfind.find(graph=graph, start=f'{mrow},{mcol}', end=f"{grow},{gcol}")
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
        s, _ = parse_state2(raw_s, maze=maze)

        while True:
            # env.render()
            me = raw_s["players"][raw_s["my_id"]]
            me_row = me["position"]["x"]
            me_col = me["position"]["y"]
            exit_row = me["exit_position"]["x"]
            exit_col = me["exit_position"]["y"]
            e_path = pathfind.find(graph=gp, start=f'{me_row},{me_col}', end=f"{exit_row},{exit_col}")
            g_path = short_gem(graph=gp, gems=raw_s["gem"], me=me)
            if len(e_path) - 1 > me["action_point"]:
                next_n = e_path[1].split(",")
            else:
                next_n = g_path[1].split(",")
            nr, nc = int(next_n[0]), int(next_n[1])
            action = move(nr, nc, me_row, me_col)
            a_int = a_map[action]
            if len(e_path) < me["action_point"]:
                x_data.append(s[None, :])
                y_data.append(a_int)
            raw_s, _, done = env.step(action)
            s, _ = parse_state2(raw_s, maze=maze)
            if done:
                break
        print(f"collect data {ep=}")
    x = np.concatenate(x_data, axis=0, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    with open(os.path.join(TMP_DIR, "mazeData.npz"), "wb") as f:
        np.savez(f, x=x, y=y)
    env.close()
    return x, y


def train_super(ep):
    trainer = build_trainer()
    data = np.load(os.path.join(TMP_DIR, "mazeData.npz"))
    trainer.train_supervised(
        x=data["x"], y=data["y"], epoch=ep, learning_rate=1e-3,
        save_dir=os.path.join(TMP_DIR, "superModel"), verbose=1)
    return trainer


def predict(load_ep):
    trainer = build_trainer()
    trainer.load_model(os.path.join(TMP_DIR, "superModel", f"ep-{load_ep}.zip"))
    env = Maze()
    for ep in range(2):
        raw_s = env.reset()
        maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        s, energy = parse_state2(raw_s, maze=maze)
        ep_r = 0
        while True:
            env.render()
            me = raw_s["players"][raw_s["my_id"]]
            me_row = me["position"]["x"]
            me_col = me["position"]["y"]
            exit_row = me["exit_position"]["x"]
            exit_col = me["exit_position"]["y"]
            path = pathfind.find(graph=gp, start=f'{me_row},{me_col}', end=f"{exit_row},{exit_col}")
            if len(path) - 1 <= energy:
                a = trainer.model.predict(s)
                action = trainer.map_action(a)
            else:
                if len(path) == 0:
                    action = "s"
                else:
                    next_n = path[1].split(",")
                    nr, nc = int(next_n[0]), int(next_n[1])
                    action = move(nr, nc, me_row, me_col)
            raw_s, r, done = env.step(action)
            ep_r += r
            s_, energy = parse_state2(raw_s, maze=maze)
            s = s_
            if done:
                break
        print(f"predict ep={ep}, ep_r={ep_r}")
    env.close()


if __name__ == "__main__":
    super_train_ep = 20
    # get_data()
    train_super(super_train_ep)
    predict(load_ep=super_train_ep)
