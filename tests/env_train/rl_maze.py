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
    trainer.set_model_encoder(pi=keras.Sequential([
        keras.layers.InputLayer((PAD_SIZE * 2 + 1) ** 2 - 1 + 8),
        keras.layers.Dense(128),
        keras.layers.ReLU(),
        # keras.layers.Dense(64),
        # keras.layers.ReLU(),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ]), critic=keras.Sequential([
        keras.layers.InputLayer((PAD_SIZE * 2 + 1) ** 2 - 1 + 8),
        keras.layers.Dense(128),
        keras.layers.ReLU(),
        # keras.layers.Dense(64),
        # keras.layers.ReLU(),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ]), action_num=4)
    trainer.set_action_transformer(
        rlearn.transformer.DiscreteAction(ACTION_ORDER)
    )
    trainer.set_params(
        learning_rate=(1e-5, 1e-4),
        batch_size=32,
        gamma=0.9,
        replace_step=500,
        epsilon_decay=1e-3,
    )
    trainer.set_replay_buffer(max_size=3000)
    return trainer


def train_rl(load_ep=None):
    env = Maze()
    trainer = build_trainer()
    if load_ep is not None:
        trainer.load_model_weights(os.path.join(TMP_DIR, "superModel", f"ep-{load_ep}.zip"))

    for ep in range(100):
        raw_s = env.reset()
        maze = pad_maze(raw_s["maze"])
        gp = get_graph(raw_s["maze"])
        s, energy = parse_state2(raw_s, maze=maze)
        step = 0
        ep_r = 0
        while True:
            env.render()
            me = raw_s["players"][raw_s["my_id"]]
            me_row = me["position"]["x"]
            me_col = me["position"]["y"]
            exit_row = me["exit_position"]["x"]
            exit_col = me["exit_position"]["y"]
            path = pathfind.find(graph=gp, start=f'{me_row},{me_col}', end=f"{exit_row},{exit_col}")
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
                    action = move(nr, nc, me_row, me_col)
            raw_s, r, done = env.step(action)
            ep_r += r
            s_, energy = parse_state2(raw_s, maze=maze)
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
    train_rl(load_ep=1)
