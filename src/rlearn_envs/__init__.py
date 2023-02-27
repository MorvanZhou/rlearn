import os

from rlearn_envs.maze.maze_env import Maze

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from rlearn_envs.base import BaseEnv
from rlearn_envs.flappy_bird import Env as FlappyBird
from rlearn_envs.junior import Env as Junior
from rlearn_envs.jumping_dino import Env as JumpingDino

GAME_MAP = {
    "flappy_bird": FlappyBird,
    "junior": Junior,
    "jumping_dino": JumpingDino,
}


def get(name: str) -> BaseEnv:
    e = GAME_MAP.get(name.lower(), None)()
    if e is None:
        raise ValueError("name of '{}' is not supported".format(name))
    return e
