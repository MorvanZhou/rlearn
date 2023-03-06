import math
import os

import numpy as np
import pathfind
from tensorflow import keras

import rlearn
from rlearn_envs.maze import Maze

TMP_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tmp", "superRlTest")
ACTION_ORDER = ["u", "d", "l", "r"]
PAD_SIZE = 2


def parse_state1(s: dict, maze: np.ndarray):
    me = s["players"][s["my_id"]]
    pos = me["position"]
    row, col = pos["x"] + PAD_SIZE, pos["y"] + PAD_SIZE
    # view: 5*5-1
    view = maze[row - PAD_SIZE: row + PAD_SIZE + 1, col - PAD_SIZE: col + PAD_SIZE + 1].ravel()
    front = (view.size - 1) // 2
    view = 1 - np.concatenate([view[:front + 1], view[front + 2:]], dtype=np.float32)
    # gd: [g0r, g0d, g1r, g1d, g2r, g2d, g3r, g3d, g4r, g4d]
    gd = np.zeros((10,), dtype=np.float32)
    gk = list(s["gem"].keys())
    gk.sort()
    for i, k in enumerate(gk):
        gdr = s["gem"][k]["y"] - pos["y"]
        gdc = s["gem"][k]["x"] - pos["x"]
        radian = np.arctan2(-gdr, gdc)  # -pi, pi
        distance_co = 1 / (0.3 + np.sqrt(np.sum(np.square([gdr, gdc]))) / 5)
        gd[i] = radian / np.pi
        gd[i + 1] = distance_co
    gd /= 15.
    state = np.concatenate([view, gd], dtype=np.float32)
    return state, me["action_point"]


def parse_state2(s: dict, maze: np.ndarray):
    # input shape: (PAD_SIZE * 2 + 1) ** 2 - 1 + 8
    me = s["players"][s["my_id"]]
    pos = me["position"]
    row, col = pos["x"] + PAD_SIZE, pos["y"] + PAD_SIZE
    # view: 5*5-1
    view = maze[row - PAD_SIZE: row + PAD_SIZE + 1, col - PAD_SIZE: col + PAD_SIZE + 1].ravel()
    front = (view.size - 1) // 2
    view = np.concatenate([view[:front + 1], view[front + 2:]], dtype=np.float32)
    # gd: [45, -45, 135, -135, 45~-45, 45~135, -45~-135, 135~180+-135~-180]
    gd = np.zeros((8,), dtype=np.float32)
    for g in s["gem"].values():
        dy = -(g["x"] - pos["x"])
        dx = g["y"] - pos["y"]
        distance_co = 1 / (1. + np.sqrt(np.sum(np.square([dx, dy]))))
        d45 = np.pi / 4
        radian = np.arctan2(dy, dx)  # -pi, pi

        if math.isclose(radian, d45, abs_tol=1e-4):
            gd[0] += distance_co
        elif math.isclose(radian, -d45, abs_tol=1e-4):
            gd[1] += distance_co
        elif math.isclose(radian, d45 * 3, abs_tol=1e-4):
            gd[2] += distance_co
        elif math.isclose(radian, -d45 * 3, abs_tol=1e-4):
            gd[3] += distance_co
        elif abs(radian) < d45:
            gd[4] += distance_co
        elif radian > d45 and radian < d45 * 3:
            gd[5] += distance_co
        elif radian < -d45 and radian > -d45 * 3:
            gd[6] += distance_co
        else:
            gd[7] += distance_co
    gd /= 5.
    state = np.concatenate([view, gd], dtype=np.float32)
    return state, me["action_point"]


def parse_state3(s: dict, maze: np.ndarray):
    # input shape: (PAD_SIZE * 2 + 1) ** 2 - 1 + 8 + 10 + 5+16
    me = s["players"][s["my_id"]]
    row, col = me.row + PAD_SIZE, me.col + PAD_SIZE
    # view: 5*5-1
    view = maze[row - PAD_SIZE: row + PAD_SIZE + 1, col - PAD_SIZE: col + PAD_SIZE + 1].ravel()
    front = (view.size - 1) // 2
    view = np.concatenate([view[:front + 1], view[front + 2:]], dtype=np.float32)
    # gd: [45, -45, 135, -135, 45~-45, 45~135, -45~-135, 135~180+-135~-180]
    gd = np.zeros((8,), dtype=np.float32)
    for g in s["gems"].values():
        g = g[0]
        dy = -(g.row - me.row)
        dx = g.col - me.col
        distance_co = 1 / (1. + np.sqrt(np.sum(np.square([dx, dy]))))
        d45 = np.pi / 4
        radian = np.arctan2(dy, dx)  # -pi, pi

        if math.isclose(radian, d45, abs_tol=1e-4):
            gd[0] += distance_co
        elif math.isclose(radian, -d45, abs_tol=1e-4):
            gd[1] += distance_co
        elif math.isclose(radian, d45 * 3, abs_tol=1e-4):
            gd[2] += distance_co
        elif math.isclose(radian, -d45 * 3, abs_tol=1e-4):
            gd[3] += distance_co
        elif abs(radian) < d45:
            gd[4] += distance_co
        elif radian > d45 and radian < d45 * 3:
            gd[5] += distance_co
        elif radian < -d45 and radian > -d45 * 3:
            gd[6] += distance_co
        else:
            gd[7] += distance_co
    gd /= 5.

    # other players:
    players = np.zeros((8,), dtype=np.float32)
    for k, p in s["players"].items():
        if k == s["my_id"]:
            continue
        dy = -(p.row - me.row)
        dx = p.col - me.col
        distance_co = 1 / (1. + np.sqrt(np.sum(np.square([dx, dy]))))
        d45 = np.pi / 4
        radian = np.arctan2(dy, dx)  # -pi, pi

        if math.isclose(radian, d45, abs_tol=1e-4):
            players[0] += distance_co
        elif math.isclose(radian, -d45, abs_tol=1e-4):
            players[1] += distance_co
        elif math.isclose(radian, d45 * 3, abs_tol=1e-4):
            players[2] += distance_co
        elif math.isclose(radian, -d45 * 3, abs_tol=1e-4):
            players[3] += distance_co
        elif abs(radian) < d45:
            players[4] += distance_co
        elif radian > d45 and radian < d45 * 3:
            players[5] += distance_co
        elif radian < -d45 and radian > -d45 * 3:
            players[6] += distance_co
        else:
            players[7] += distance_co
    players /= 5.

    # keys = list(me.item_count.keys())
    # item_collection = np.zeros((len(keys),), dtype=np.float32)
    # keys.sort()
    # min_g_n = 0
    # for mgk in keys:
    #     if me["gem"][mgk] < min_g_n:
    #         min_g_n = me["gem"][mgk]
    # for n in range(len(keys)):
    #     item_collection[n] = me["gem"][keys[n]] - min_g_n
    #
    # # gd: [g0r, g0d, g1r, g1d, g2r, g2d, g3r, g3d, g4r, g4d]
    # coll_data = np.zeros((10,), dtype=np.float32)
    # gk = list(s["gem"].keys())
    # gk.sort()
    # for i, k in enumerate(gk):
    #     if item_collection[i] != 0:
    #         continue
    #     gdr = s["gem"][k]["y"] - pos["y"]
    #     gdc = s["gem"][k]["x"] - pos["x"]
    #     radian = np.arctan2(-gdr, gdc)  # -pi, pi
    #     distance_co = 1 / (1 + np.sqrt(np.sum(np.square([gdr, gdc]))))
    #     coll_data[i*2] = radian / np.pi
    #     coll_data[i*2 + 1] = distance_co
    # coll_data /= 15.

    state = np.concatenate([
        view,
        gd,
        players,
        # item_collection,
        # coll_data
    ], dtype=np.float32)

    return state, me.energy


def pad_maze(maze, d=2):
    m = np.ones((maze.shape[0] + d * 2, maze.shape[1] + d * 2), dtype=np.float32)
    m[d:-d, d:-d] = maze
    return m


def move(nr, nc, mr, mc):
    if nr == mr:
        if nc > mc:
            return "r"
        else:
            return "l"
    else:
        dr = nr - mr
        if dr == 1:
            return "d"
        elif dr == 0:
            return "s"
        else:
            return "u"


def get_graph(maze):
    _COST_MAP = {
        1: pathfind.INFINITY,
        0: 1,
    }
    matrix = []
    for row in maze:
        new_row = []
        for cell in row:
            cost = _COST_MAP[cell]
            new_row.append(cost)
        matrix.append(new_row)
    return pathfind.transform.matrix2graph(matrix, diagonal=False)


def build_trainer():
    tmp = os.path.join(os.path.dirname(__file__), "tmp")
    trainer = rlearn.PPODiscreteTrainer(log_dir=tmp)
    input_size = (PAD_SIZE * 2 + 1) ** 2 - 1 + 8 + 8  # + 10 + 5
    trainer.set_model_encoder(pi=keras.Sequential([
        keras.layers.InputLayer(input_size),
        keras.layers.Dense(256),
        keras.layers.ReLU(),
        keras.layers.Dense(128),
        keras.layers.ReLU(),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ]), critic=keras.Sequential([
        keras.layers.InputLayer(input_size),
        keras.layers.Dense(256),
        keras.layers.ReLU(),
        keras.layers.Dense(128),
        keras.layers.ReLU(),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ]), action_num=4)
    trainer.set_action_transformer(
        rlearn.transformer.DiscreteAction(ACTION_ORDER)
    )
    trainer.set_params(
        learning_rate=(1e-4, 1e-3),
        batch_size=32,
        gamma=0.9,
        replace_step=1000,
        epsilon_decay=1e-3,
    )
    trainer.set_replay_buffer(max_size=50000)
    return trainer


def train_rl(load_ep=None):
    env = Maze()
    trainer = build_trainer()
    if load_ep is not None:
        rlearn.supervised.set_actor_weights(trainer, os.path.join(TMP_DIR, "superModel", f"ep-{load_ep}.zip"))

    for ep in range(400):
        raw_s = env.reset()
        maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        s, energy = parse_state3(raw_s, maze=maze)
        step = 0
        ep_r = 0
        while True:
            if args.display:
                env.render()
            me = raw_s["players"][raw_s["my_id"]]
            exit = raw_s["exits"][raw_s["my_id"]]
            path = pathfind.find(graph=gp, start=f'{me.row},{me.col}', end=f"{exit.row},{exit.col}")
            a = -1
            if len(path) - 1 <= energy:
                a = trainer.predict(s)
                action = trainer.map_action(a)
            else:
                if len(path) == 0:
                    action = "s"
                else:
                    next_n = path[1].split(",")
                    nr, nc = int(next_n[0]), int(next_n[1])
                    action = move(nr, nc, me.row, me.col)
            raw_s, r, done = env.step(action)
            ep_r += r
            s_, energy = parse_state3(raw_s, maze=maze)
            if a != -1:
                trainer.store_transition(s=s, a=a, r=r, s_=s_, done=done)
                trainer.train_batch()
            s = s_
            if done:
                break

            step += 1
        trainer.save_model_weights(os.path.join(TMP_DIR, "rlModel", f"ep-{ep + 1}.zip"))
        print(f"{ep=}, {ep_r=:.3f}, {trainer.epsilon=:.3f}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("-l", "--load", type=int, default=0)
    args = parser.parse_args()
    if not args.display:
        os.environ['SDL_VIDEODRIVER'] = "dummy"
    if args.load == 0:
        train_rl(load_ep=None)
    else:
        train_rl(load_ep=args.load)
