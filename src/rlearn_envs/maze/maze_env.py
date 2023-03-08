import copy
import json
import math
import os
import random
import time
import typing as tp
from dataclasses import dataclass

import numpy as np
import pygame

from rlearn import EnvWrapper, State
from rlearn_envs.maze import load


def game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


@dataclass
class Player:
    """
    Player 类，包含用户 id，行列坐标，当前朝向，体力

    Attributes:
    id (int): 用户 id，全局唯一
    row (int): 行坐标，从 0 开始计数
    col (int): 列坐标，从 0 开始计数
    direction (int): 当前朝向，为 ["U", "D", "L", "R"] 中的一种
    energy (float): 当前体力
    score (float): 当前得分
    finished (bool): 是否已结束
    item_count (dict[str, int]): 收集到的宝石和宝箱计数
    """
    id: int
    row: int
    col: int
    direction: str
    energy: float
    score: float
    finished: bool
    item_count: tp.Dict[str, int]


@dataclass
class Item:
    """
    Item 类，可访问具体物品的行列坐标

    Attributes:
    row (int): 行坐标，从 0 开始计数
    col (int): 列坐标，从 0 开始计数
    """
    row: int
    col: int


class Maze(EnvWrapper):
    def __init__(self):
        super().__init__()
        game_dir = os.path.dirname(__file__)
        self._map_data = self.load(os.path.join(game_dir, "data", "map.json"))
        self.board = np.array(self._map_data.get("mapData", [[]]))
        self.copy_board = copy.deepcopy(self.board)
        self.row = self.board.shape[0]
        self.col = self.board.shape[1]
        # 获取地图网格信息
        if self.row == 0 or self.col == 0:
            raise ValueError("地图网格数据异常，请检查json文件中的mapData参数！")
        # 获取玩家信息
        self.players_info = self._map_data.get("players", [])
        self.players_num = len(self.players_info)
        if self.players_num == 0:
            raise ValueError("玩家数量不能少于1个，请检查json文件中的players参数！")
        # 获取终点信息
        self.exits_info = self._map_data.get("exits", [])
        if len(self.exits_info) != self.players_num:
            raise ValueError("终点数量和玩家数量不匹配，请检查json文件中的exits参数！")
        self.players_dict: tp.Dict[int, Player] = dict()
        self.exits_dict = dict()
        # 设定先手玩家
        self.cur_player = 0
        # 获取移动损耗
        self.map_weights = dict()
        map_weights = self._map_data.get("mapWeights", None)
        if not map_weights:
            self.map_weights["move"] = 1
            self.map_weights["stay"] = 1
        else:
            self.map_weights["move"] = map_weights.get("move", {"0": 1}).get("0", 1)
            self.map_weights["stay"] = map_weights.get("stay", {"0": 1}).get("0", 1)
        # 获取宝石信息
        self.gem_type = {101: "pink_gem", 102: "red_gem", 103: "yellow_gem", 104: "purple_gem",
                         105: "blue_gem", 201: "box"}
        self.gem_info = self._map_data.get("items", [])
        if len(self.gem_info) > 6:
            raise ValueError("items数量不能大于6个，请检查json文件中的items参数")
        self.gem_dict: tp.Dict[str, tp.List[Item]] = dict()
        self.collections_config = self._map_data.get("collectionsConfig")
        self.effect_value = self.collections_config[0].get("effectValues", [10])[0]
        self.action_dict = {
            "u": [-1, 0],
            "d": [1, 0],
            "l": [0, -1],
            "r": [0, 1],
            "s": [0, 0]
        }
        self.exits_pos_set = set()

        # 用于render渲染
        pygame.init()
        self.FPS_CLOCK = pygame.time.Clock()
        self.screen_size = (1050, 1400)
        self.screen_size = (200 + 1000 + 200, 1000)
        self.limit_distance_x = int(self.screen_size[1] / self.row)
        self.limit_distance_y = int(self.screen_size[1] / self.col)
        self.screen_size = (200 + self.limit_distance_x * self.row + 200, self.limit_distance_y * self.col)
        self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
        pygame.display.set_caption("maze")
        self.viewer.fill(pygame.Color("white"))
        # 加载地图元素
        self.bush = load.load_img("bush")
        self.grass = load.load_img("grass")
        self.land = load.load_img("land")
        self.stone = load.load_img("stone")
        self.tree = load.load_img("tree")
        self.water1 = load.load_img("water1")
        self.water2 = load.load_img("water2")
        self.wood1 = load.load_img("wood1")
        self.wood2 = load.load_img("wood2")
        # 加载玩家元素
        self.blue_player = load.load_img("blue_player")
        self.green_player = load.load_img("green_player")
        self.red_player = load.load_img("red_player")
        self.yellow_player = load.load_img("yellow_player")
        # 加载终点元素
        self.blue_exits = load.load_img("blue_exits")
        self.green_exits = load.load_img("green_exits")
        self.red_exits = load.load_img("red_exits")
        self.yellow_exits = load.load_img("yellow_exits")
        # 加载宝石图片
        self.pink_gem = load.load_img("pink_gem")
        self.red_gem = load.load_img("red_gem")
        self.blue_gem = load.load_img("blue_gem")
        self.yellow_gem = load.load_img("yellow_gem")
        self.purple_gem = load.load_img("purple_gem")
        self.bonus = load.load_img("bonus")
        self.box = load.load_img("box")
        self.players_bonus: tp.Dict[int, int] = dict()
        self.players_exit: tp.Dict[int, Item] = dict()

        self.map_param = [self.bush, self.grass, self.tree, self.stone,
                          self.water1, self.water2, self.wood1, self.wood2]
        self.player_param = [self.blue_player, self.green_player, self.yellow_player, self.red_player]
        self.player_queue = ["blue", "green", "yellow", "red"]
        self.exits_param = [self.blue_exits, self.green_exits, self.yellow_exits, self.red_exits]
        self.gem_param = {"red_gem": self.red_gem,
                          "blue_gem": self.blue_gem,
                          "yellow_gem": self.yellow_gem,
                          "pink_gem": self.pink_gem,
                          "purple_gem": self.purple_gem,
                          "box": self.box}

        self.render_board = np.multiply(np.floor(np.random.rand(self.row, self.col) * len(self.map_param)), self.board)
        pygame.draw.lines(self.viewer, (0, 0, 0), True,
                          ((0, self.screen_size[1]/2),
                           (1400, self.screen_size[1]/2)), 5)

        for x in range(self.row):
            for y in range(self.col):
                if self.board[x][y] == 0:
                    self.viewer.blit(
                        pygame.transform.scale(self.land, (self.limit_distance_y, self.limit_distance_x)),
                        (200 + y * self.limit_distance_y, x * self.limit_distance_x))
                else:
                    self.viewer.blit(
                        pygame.transform.scale(
                            self.map_param[int(self.render_board[x][y])],
                            (self.limit_distance_y, self.limit_distance_x)),
                        (200 + y * self.limit_distance_y, x * self.limit_distance_x))

    def reset(self) -> State:
        # 存储玩家id值，避免出现重复id
        exist_id = set()
        # 默认先手玩家为id = 0的玩家
        self.cur_player = 0
        self.players_bonus = {}
        self.players_exit = {}
        self.copy_board = copy.deepcopy(self.board)
        self.exits_pos_set = set()
        # 重置玩家初始信息
        for player, exits in zip(self.players_info, self.exits_info):
            player_id = player.get("id", None)
            if player_id is None or not isinstance(player_id, int):
                raise ValueError("玩家id不能为空，请检查json文件中的players参数！")
            if player_id in exist_id:
                raise ValueError("玩家id存在重复，请检查json文件中的players参数！")
            else:
                exist_id.add(player_id)
            player_position = player.get("position", None)
            if not player_position:
                raise ValueError("玩家初始坐标值不能为空，请检查json文件中的players参数！")
            if player_position["x"] < 0 or player_position["x"] >= self.row or player_position["y"] < 0 or \
                    player_position["y"] >= self.col:
                raise ValueError("玩家初始坐标值不在地图范围内，请检查json文件中的players参数！")
            player_action_point = player.get("actionPoints", None)
            if not isinstance(player_action_point, int) or player_action_point <= 0:
                raise ValueError("玩家活动点数必须为大于0的int类型数值，请检查json文件中的players参数！")
            exit_position = exits.get("position", None)
            if not exit_position:
                raise ValueError("终点初始坐标值不能为空，请检查json文件中的exits参数！")
            if exit_position["x"] < 0 or exit_position["x"] >= self.row or exit_position["y"] < 0 or \
                    exit_position["y"] >= self.col:
                raise ValueError("终点初始坐标值不在地图范围内，请检查json文件中的exits参数！")
            self.exits_pos_set.add((exit_position["x"], exit_position["y"]))
            self.players_dict[player_id] = Player(
                id=player_id,
                row=player_position["x"],
                col=player_position["y"],
                energy=player_action_point,
                direction="d",
                score=0,
                finished=False,
                item_count={t: 0 for t in self.gem_type.values()}
            )
            self.players_bonus[player_id] = 0
            self.players_exit[player_id] = Item(row=exit_position["x"], col=exit_position["y"])

            self.copy_board[player_position["x"]][player_position["y"]] = 1
            self.copy_board[exit_position["x"]][exit_position["y"]] = 1

        for gem_id in range(len(self.gem_info)):
            gem_info = self.gem_info[gem_id]
            gem_type = self.gem_type[gem_info["objectType"]]
            gem_position = gem_info.get("position", None)
            if not gem_position:
                raise ValueError("玩家初始坐标值不能为空，请检查json文件中的players参数！")
            if gem_position["x"] < 0 or gem_position["x"] >= self.row or gem_position["y"] < 0 or \
                    gem_position["y"] >= self.col:
                raise ValueError("宝石/宝箱初始坐标值不在地图范围内，请检查json文件中的items参数！")
            self.gem_dict[gem_type] = [Item(row=gem_position["x"], col=gem_position["y"])]
            self.copy_board[gem_position["x"]][gem_position["y"]] = 1
        for index, player in enumerate(self.players_dict):
            p = self.players_dict[player]
            player_x = p.row
            player_y = p.col
            player_img = self.player_param[index]
            player_img = pygame.transform.scale(player_img, (self.limit_distance_y, self.limit_distance_x))
            player_rect = player_img.get_rect()
            player_rect.x = 200 + player_y * self.limit_distance_y
            player_rect.y = player_x * self.limit_distance_x
            print(player_rect)
            self.viewer.blit(player_img, player_rect)
            # self.viewer.blit(pygame.transform.scale(
            #     self.player_param[index], (self.limit_distance_y, self.limit_distance_x)).get_rect(),
            #     (200 + player_y * self.limit_distance_y, player_x * self.limit_distance_x))
            # background = pygame.Surface((self.limit_distance_x, self.limit_distance_y))
            # background.set_alpha(1)
            # background.fill(pygame.Color("white"))
            # background.set_alpha(1)
            # background.blit(pygame.transform.scale(
            #     self.player_param[index], (self.limit_distance_y, self.limit_distance_x)), (0, 0))
            # self.viewer.blit(background, (200 + player_y * self.limit_distance_y, player_x * self.limit_distance_x))
        return {
            "players": self.players_dict,
            "gems": self.gem_dict,
            "maze": self.board,
            "my_id": self.cur_player,
            "exits": self.players_exit,
            "collected": "",
        }

    def step(self, action):
        reward = -0.01
        finish = False
        collected = ""
        player = self.players_dict[self.cur_player]
        # 当玩家行动点为零且游戏未结束（部分用户吃到宝箱可能导致行动点增加或减少），用户的action设置为"s"且该行为不会影响用户的行动点数值
        if action == "s":
            if player.energy >= self.map_weights["stay"]:
                player.energy -= self.map_weights["stay"]
        else:
            if player.energy >= self.map_weights["move"]:
                player.energy -= self.map_weights["move"]
            else:
                action = "s"
        move_data = self.action_dict[action]
        target_x = player.row + move_data[0]
        target_y = player.col + move_data[1]
        if 0 <= target_x < self.row and 0 <= target_y < self.col and self.board[target_x][target_y] == 0:
            if sum(list(1 if p.row == player.row and p.col == player.col else 0 for p in
                        self.players_dict.values())) == 1:
                self.copy_board[player.row][player.col] = 0
            player.row = target_x
            player.col = target_y
            self.copy_board[target_x][target_y] = 1
            for gem_name, gem_list in self.gem_dict.items():
                gem = gem_list[0]
                gem_pos_x = gem.row
                gem_pos_y = gem.col
                if gem_pos_x == target_x and gem_pos_y == target_y:
                    collected = gem_name
                    player.item_count[gem_name] += 1
                    if gem_name == "box":
                        # todo
                        print("this is box, do nothing!")
                    else:
                        reward += 1
                        player.score += self.effect_value
                    while self.copy_board[gem_pos_x][gem_pos_y] != 0 or (gem_pos_x, gem_pos_y) in self.exits_pos_set:
                        gem_pos_x = math.floor(random.random() * self.row)
                        gem_pos_y = math.floor(random.random() * self.col)
                    self.copy_board[gem_pos_x][gem_pos_y] = 1
                    gem.row = gem_pos_x
                    gem.col = gem_pos_y
                    if len(list(
                            filter(lambda x:
                                   player.item_count[x] <=
                                   self.players_bonus[player.id],
                                   [k for k in player.item_count.keys() if k.endswith("_gem")]))) == 0:
                        self.players_bonus[player.id] += 1
                        reward += 5
                        player.score += 5
                    break
        if len(list(filter(
                lambda x: self.players_dict[x].energy > self.map_weights["stay"] and self.players_dict[x].energy >
                          self.map_weights["move"], self.players_dict))) == 0:
            finish = True
            reward = -1
        elif all([(p.row == self.players_exit[p.id].row) and (p.col == self.players_exit[p.id].col)
                  for p in self.players_dict.values()]):
            finish = True

        self.cur_player = (self.cur_player + 1) % self.players_num
        return {
                   "players": self.players_dict,
                   "gems": self.gem_dict,
                   "maze": self.board,
                   "my_id": self.cur_player,
                   "exits": self.players_exit,
                   "collected": collected,
               }, reward, finish

    def render(self):
        gem_width = 20
        font_size = 20
        # pygame.display.set_caption("maze")
        # self.viewer.fill(pygame.Color("white"))

        # 画直线
        # pygame.draw.lines(self.viewer, (0, 0, 0), True,
        #                   ((0, self.screen_size[1]/2),
        #                    (1400, self.screen_size[1]/2)), 5)
        #
        # for x in range(self.row):
        #     for y in range(self.col):
        #         if self.board[x][y] == 0:
        #             self.viewer.blit(
        #                 pygame.transform.scale(self.land, (self.limit_distance_y, self.limit_distance_x)),
        #                 (200 + y * self.limit_distance_y, x * self.limit_distance_x))
        #         else:
        #             self.viewer.blit(
        #                 pygame.transform.scale(
        #                     self.map_param[int(self.render_board[x][y])],
        #                     (self.limit_distance_y, self.limit_distance_x)),
        #                 (200 + y * self.limit_distance_y, x * self.limit_distance_x))

        for index, player in enumerate(self.players_dict):
            basic_x = 20 + math.floor(index / 2) * 1200
            basic_y = 20 + index % 2 * 500
            exits_x = self.players_exit[index].row
            exits_y = self.players_exit[index].col
            self.viewer.blit(pygame.transform.scale(
                self.exits_param[index], (self.limit_distance_y, self.limit_distance_x)),
                (200 + exits_y * self.limit_distance_y, exits_x * self.limit_distance_x))
            p = self.players_dict[player]
            player_x = p.row
            player_y = p.col
            # self.viewer.blit(pygame.transform.scale(
            #     self.player_param[index], (self.limit_distance_y, self.limit_distance_x)),
            #     (200 + player_y * self.limit_distance_y, player_x * self.limit_distance_x))
            font = pygame.font.SysFont('inkfree', font_size)

            text = font.render(self.player_queue[index] + " id:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 20))
            text = font.render(self.player_queue[index] + " id:", True, (0, 0, 0))
            self.viewer.blit(text, (0 + basic_x, 0 + basic_y))

            text = font.render(str(index), True, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 20))
            text = font.render(str(index), True, (0, 0, 0))
            self.viewer.blit(text, (120 + basic_x, 0 + basic_y))

            text = font.render(self.player_queue[index] + " score:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 55))
            text = font.render(self.player_queue[index] + " score:", True, (0, 0, 0))
            self.viewer.blit(text, (0 + basic_x, 35 + basic_y))

            text = font.render(str(p.score), True, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 55))
            text = font.render(str(p.score), True, (0, 0, 0))
            self.viewer.blit(text, (120 + basic_x, 35 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (120 + basic_x, 35 + basic_y))

            text = font.render(self.player_queue[index] + " step:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 90))
            text = font.render(self.player_queue[index] + " step:", True, (0, 0, 0))
            self.viewer.blit(text, (0 + basic_x, 70 + basic_y))

            text = font.render(str(p.energy), True, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 90))
            text = font.render(str(p.energy), True, (0, 0, 0))
            self.viewer.blit(text, (120 + basic_x, 70 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (120 + basic_x, 70 + basic_y))

            # 绘制玩家获取的宝石数量
            self.viewer.blit(pygame.transform.scale(self.pink_gem, (gem_width, gem_width)),
                             (0 + basic_x, 105 + basic_y))
            text = font.render(" ×   " + str(p.item_count["pink_gem"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 105 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 105 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.red_gem, (gem_width, gem_width)),
                             (0 + basic_x, 140 + basic_y))
            text = font.render(" ×   " + str(p.item_count["red_gem"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 140 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 140 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.yellow_gem, (gem_width, gem_width)),
                             (0 + basic_x, 175 + basic_y))
            text = font.render(" ×   " + str(p.item_count["yellow_gem"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 175 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 175 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.blue_gem, (gem_width, gem_width)),
                             (0 + basic_x, 210 + basic_y))
            text = font.render(" ×   " + str(p.item_count["blue_gem"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 210 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 210 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.purple_gem, (gem_width, gem_width)),
                             (0 + basic_x, 245 + basic_y))
            text = font.render(" ×   " + str(p.item_count["purple_gem"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 245 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 245 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.box, (gem_width, gem_width)),
                             (0 + basic_x, 280 + basic_y))
            text = font.render(" ×   " + str(p.item_count["box"]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 280 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 280 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.bonus, (gem_width, gem_width)),
                             (0 + basic_x, 315 + basic_y))
            text = font.render(" ×   " + str(self.players_bonus[player]), True, (0, 0, 0))
            self.viewer.blit(text, (30 + basic_x, 315 + basic_y))
            background = pygame.Surface((60, 40))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(background, (30 + basic_x, 315 + basic_y))
        for gem in self.gem_dict:
            g = self.gem_dict[gem][0]
            gem_x = g.row
            gem_y = g.col
            self.viewer.blit(
                pygame.transform.scale(self.gem_param[gem], (self.limit_distance_y, self.limit_distance_x)),
                (200 + gem_y * self.limit_distance_y, gem_x * self.limit_distance_x))

        pygame.display.update()
        game_over()
        time.sleep(0)  # 控制每帧渲染持续时间
        self.FPS_CLOCK.tick(400)  # 控制刷新速度，值越大刷新越快

    def load(self, map_json: tp.Any):
        with open(map_json) as file:
            map_data = json.load(file)
            return map_data

    @staticmethod
    def close():
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    maze = Maze()
    actions = ["u", "d", "l", "r", "s"]
    for _ in range(10000):
        maze.reset()
        while True:
            state, reward, done = maze.step(actions[math.floor(random.random() * 5)])
            maze.render()
            if done:
                break
