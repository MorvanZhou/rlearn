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


class Sprite(pygame.sprite.Sprite):
    def __init__(self, img, x, y):
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self) -> None:
        pass

    def move(self, x, y):
        self.rect.x = x
        self.rect.y = y


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
    def __init__(self, screen_width=500, screen_height=500):
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
        self.items_type = {101: "red_gem", 102: "pink_gem", 103: "blue_gem", 104: "yellow_gem",
                           105: "purple_gem", 201: "box"}
        self.items_info = self._map_data.get("items", [])
        if len(self.items_info) > 6:
            raise ValueError("items数量不能大于6个，请检查json文件中的items参数")
        self.items_dict: tp.Dict[str, tp.List[Item]] = dict()
        self.collections_config = self._map_data.get("collectionsConfig")
        self.effect_value = self.collections_config[0].get("effectValues", [10])[0]
        self.action_dict = {
            "u": [-1, 0],
            "d": [1, 0],
            "l": [0, -1],
            "r": [0, 1],
            "s": [0, 0]
        }
        self.exits_pos_set = list()

        # 用于render渲染
        pygame.init()
        self.FPS_CLOCK = pygame.time.Clock()
        # 控制渲染中间部分的地图范围
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 控制渲染两侧的用户信息范围
        self.screen_pad = 200
        self.screen_size = (self.screen_pad + self.screen_height + self.screen_pad, self.screen_width)
        # 根据地图行列数动态控制地图中每个网格的尺寸
        self.limit_distance_x = int(self.screen_size[1] / self.row)
        self.limit_distance_y = int(self.screen_size[1] / self.col)
        if self.limit_distance_x > self.limit_distance_y:
            self.limit_distance_x = self.limit_distance_y
        if self.limit_distance_y > self.limit_distance_x:
            self.limit_distance_y = self.limit_distance_x
        # 处理动态获取每个网格尺寸时不能整除导致的细微尺寸偏差
        self.screen_size = (self.screen_pad +
                            self.limit_distance_x * self.col + self.screen_pad, self.limit_distance_y * self.row)
        self.screen_width = self.screen_size[1]
        self.screen_height = self.screen_size[0]
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

        # 地图背景网格元素
        self.map_param = [self.bush, self.grass, self.tree, self.stone,
                          self.water1, self.water2, self.wood1, self.wood2]
        self.player_param = [self.blue_player, self.green_player, self.yellow_player, self.red_player]
        self.player_queue = ["blue", "green", "yellow", "red"]
        self.exits_param = [self.blue_exits, self.green_exits, self.yellow_exits, self.red_exits]
        self.items_param = {
            "red_gem": self.red_gem,
            "blue_gem": self.blue_gem,
            "yellow_gem": self.yellow_gem,
            "pink_gem": self.pink_gem,
            "purple_gem": self.purple_gem,
            "box": self.box
        }

        pygame.draw.lines(self.viewer, (0, 0, 0), True,
                          ((0, self.screen_size[1] / 2),
                           (self.screen_size[0], self.screen_size[1] / 2)), 5)

        # 用于保存地图背景网格的group
        self.map_group = pygame.sprite.Group()
        for x in range(self.row):
            for y in range(self.col):
                if self.board[x][y] == 0:
                    map_img = pygame.transform.scale(self.land, (self.limit_distance_y, self.limit_distance_x))
                else:
                    map_img = pygame.transform.scale(
                        self.map_param[math.floor(random.random() * len(self.map_param))],
                        (self.limit_distance_y, self.limit_distance_x))
                bg_sprite = Sprite(map_img, self.screen_pad + y * self.limit_distance_y, x * self.limit_distance_x)
                self.map_group.add(bg_sprite)

        self.font_size = int(25 * self.screen_width / 1000)  # 展示用户信息的字体的尺寸
        self.gem_width = 20 * self.screen_width / 1000  # 展示用户信息中宝石的尺寸
        self.font = pygame.font.SysFont("arial", self.font_size)
        # 配置render过程中不会变化的元素
        for index in range(self.players_num):
            # 根据用户的id获取用户信息的展示区域
            basic_x = 20 * self.screen_width / 1000 + math.floor(index / 2) * (self.screen_width + self.screen_pad)
            basic_y = 20 * self.screen_width / 1000 + index % 2 * self.screen_width / 2
            # 设置用户id的展示位置
            text = self.font.render(self.player_queue[index] + " id:", True, (0, 0, 0))
            self.viewer.blit(text, (0 * self.screen_width / 1000 + basic_x, 0 * self.screen_width / 1000 + basic_y))
            text = self.font.render(str(index), True, (0, 0, 0))
            self.viewer.blit(text, (120 * self.screen_width / 1000 + basic_x, 0 * self.screen_width / 1000 + basic_y))
            # 设置用户分数score的展示位置
            text = self.font.render(self.player_queue[index] + " score:", True, (0, 0, 0))
            self.viewer.blit(text, (0 * self.screen_width / 1000 + basic_x, 40 * self.screen_width / 1000 + basic_y))
            text = self.font.render(str(0), True, (0, 0, 0))
            self.viewer.blit(text, (120 * self.screen_width / 1000 + basic_x, 40 * self.screen_width / 1000 + basic_y))
            # 设置用户剩余活动点step的展示位置
            text = self.font.render(self.player_queue[index] + " step:", True, (0, 0, 0))
            self.viewer.blit(text, (0 * self.screen_width / 1000 + basic_x, 80 * self.screen_width / 1000 + basic_y))
            text = self.font.render(str(0), True, (0, 0, 0))
            self.viewer.blit(text, (120 * self.screen_width / 1000 + basic_x, 80 * self.screen_width / 1000 + basic_y))

            # 绘制玩家获取的宝石数量
            self.viewer.blit(pygame.transform.scale(self.pink_gem, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 125 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 120 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.red_gem, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 170 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 165 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.yellow_gem, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 215 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 210 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.blue_gem, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 260 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 255 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.purple_gem, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 305 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 300 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.box, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 350 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 345 * self.screen_width / 1000 + basic_y))

            self.viewer.blit(pygame.transform.scale(self.bonus, (self.gem_width, self.gem_width)),
                             (0 * self.screen_width / 1000 + basic_x, 395 * self.screen_width / 1000 + basic_y))
            text = self.font.render(" ×   " + str(0), True, (0, 0, 0))
            self.viewer.blit(text, (30 * self.screen_width / 1000 + basic_x, 390 * self.screen_width / 1000 + basic_y))

        # 用于保存用户sprite的group
        self.players_group = pygame.sprite.Group()
        # 用于保存宝石sprite的group
        self.gems_group = pygame.sprite.Group()
        # 用于保存用户终点sprite的group
        self.exits_group = pygame.sprite.Group()
        # 用于保存用户sprite的列表
        self.players_list = []
        # 用于保存宝石sprite的列表
        self.gems_list = []

    def reset(self) -> State:
        # 存储玩家id值，避免出现重复id
        exist_id = set()
        # 默认先手玩家为id = 0的玩家
        self.cur_player = 0
        self.players_bonus = {}
        self.players_exit = {}
        self.copy_board = copy.deepcopy(self.board)
        self.exits_pos_set = list()
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
            self.exits_pos_set.append([exit_position["x"], exit_position["y"]])

            self.players_dict[player_id] = Player(
                id=player_id,
                row=player_position["x"],
                col=player_position["y"],
                energy=player_action_point,
                direction="d",
                score=0,
                finished=False,
                item_count={t: 0 for t in self.items_type.values()}
            )
            self.players_bonus[player_id] = 0
            self.players_exit[player_id] = Item(row=exit_position["x"], col=exit_position["y"])

            self.copy_board[player_position["x"]][player_position["y"]] = 1
            self.copy_board[exit_position["x"]][exit_position["y"]] = 1

        for iid in range(len(self.items_info)):
            iinfo = self.items_info[iid]
            itype = self.items_type[iinfo["objectType"]]
            iposition = iinfo.get("position", None)
            if not iposition:
                raise ValueError("玩家初始坐标值不能为空，请检查json文件中的players参数！")
            ix = iposition["x"]
            iy = iposition["y"]
            if ix < 0 or ix >= self.row or iy < 0 or iy >= self.col:
                raise ValueError("宝石/宝箱初始坐标值不在地图范围内，请检查json文件中的items参数！")
            self.items_dict[itype] = [Item(row=iposition["x"], col=iposition["y"])]
            self.copy_board[iposition["x"]][iposition["y"]] = 1
            self.items_dict[itype] = [Item(row=ix, col=iy)]
            self.copy_board[ix][iy] = 1
            gem_img = self.items_param[itype]
            gem_img = pygame.transform.scale(gem_img, (self.limit_distance_y, self.limit_distance_x))
            gem_sprite = Sprite(gem_img, self.screen_pad + iy * self.limit_distance_y, ix * self.limit_distance_x)
            self.gems_list.append(gem_sprite)
            self.gems_group.add(gem_sprite)
            self.gems_group.draw(self.viewer)

        for index, player in enumerate(self.players_dict):
            p = self.players_dict[player]
            player_x = p.row
            player_y = p.col

            player_img = self.player_param[index]
            player_img = pygame.transform.scale(player_img, (self.limit_distance_y, self.limit_distance_x))
            player_sprite = Sprite(player_img, self.screen_pad + player_y * self.limit_distance_y,
                                   player_x * self.limit_distance_x)
            self.players_list.append(player_sprite)
            self.players_group.add(player_sprite)
            self.players_group.draw(self.viewer)

        for index, exits in enumerate(self.exits_pos_set):
            exits_img = self.exits_param[index]
            exits_x = exits[0]
            exits_y = exits[1]
            exits_img = pygame.transform.scale(exits_img, (self.limit_distance_y, self.limit_distance_x))
            exits_sprite = Sprite(exits_img, self.screen_pad + exits_y * self.limit_distance_y,
                                  exits_x * self.limit_distance_x)
            self.exits_group.add(exits_sprite)
            self.exits_group.draw(self.viewer)
        return {
            "players": self.players_dict,
            "items": self.items_dict,
            "maze": self.board,
            "my_id": self.cur_player,
            "exits": self.players_exit,
            "collected": "",
        }

    def step(self, action):
        action = action.lower()
        finish = False
        collected = ""
        player = self.players_dict[self.cur_player]
        # 当玩家行动点为零且游戏未结束（部分用户吃到宝箱可能导致行动点增加或减少），用户的action设置为"s"且该行为不会影响用户的行动点数值
        if action == "s":
            de = self.map_weights["stay"]
        else:
            de = self.map_weights["move"]
        player.energy = max(player.energy - de, 0)
        reward = -de * 0.01

        move_data = self.action_dict[action]
        target_x = player.row + move_data[0]
        target_y = player.col + move_data[1]

        hit_wall_penalty = -0.05
        try:
            tile_type = self.board[target_x][target_y]
        except IndexError:
            reward += hit_wall_penalty  # run into wall
        else:
            # run into wall
            if tile_type != 0:
                reward += hit_wall_penalty

        if 0 <= target_x < self.row and 0 <= target_y < self.col and self.board[target_x][target_y] == 0:
            if sum(list(1 if p.row == player.row and p.col == player.col else 0 for p in
                        self.players_dict.values())) == 1:
                self.copy_board[player.row][player.col] = 0
            player.row = target_x
            player.col = target_y
            self.copy_board[target_x][target_y] = 1
            for i_name, i_list in self.items_dict.items():
                item = i_list[0]
                item_x = item.row
                item_y = item.col
                if item_x == target_x and item_y == target_y:
                    collected = i_name
                    player.item_count[i_name] += 1
                    if i_name == "box":
                        # todo
                        pass
                        # print("this is box, do nothing!")
                    else:
                        reward = 1
                        player.score += self.effect_value
                    while self.copy_board[item_x][item_y] != 0 or [item_x, item_y] in self.exits_pos_set:
                        item_x = math.floor(random.random() * self.row)
                        item_y = math.floor(random.random() * self.col)
                    self.copy_board[item_x][item_y] = 1
                    item.row = item_x
                    item.col = item_y
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
            "items": self.items_dict,
            "maze": self.board,
            "my_id": self.cur_player,
            "exits": self.players_exit,
            "collected": collected,
        }, reward, finish

    def render(self):
        self.players_group.empty()
        for index, player in enumerate(self.players_dict):
            basic_x = 20 * self.screen_width / 1000 + math.floor(index / 2) * (self.screen_width + self.screen_pad)
            basic_y = 20 * self.screen_width / 1000 + index % 2 * self.screen_width / 2
            p = self.players_dict[player]
            player_x = p.row
            player_y = p.col
            self.players_list[index].move(self.screen_pad + player_y * self.limit_distance_y,
                                          player_x * self.limit_distance_x)
            self.players_group.add(self.players_list[index])

            text = self.font.render(str(p.score), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (120 * self.screen_width / 1000 + basic_x, 40 * self.screen_width / 1000 + basic_y))

            text = self.font.render(str(p.energy), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (120 * self.screen_width / 1000 + basic_x, 80 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["pink_gem"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 120 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["red_gem"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 165 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["yellow_gem"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 210 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["blue_gem"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 255 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["purple_gem"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 300 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(p.item_count["box"]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 345 * self.screen_width / 1000 + basic_y))

            text = self.font.render(" ×   " + str(self.players_bonus[player]), True, (0, 0, 0))
            background = pygame.Surface((100 * self.screen_width / 1000, 50 * self.screen_width / 1000))
            background.fill(pygame.Color("white"))
            background.blit(text, (0, 0))
            self.viewer.blit(
                background,
                (30 * self.screen_width / 1000 + basic_x, 390 * self.screen_width / 1000 + basic_y))

        self.gems_group.empty()
        for index, gem in enumerate(self.items_dict):
            g = self.items_dict[gem][0]
            gem_x = g.row
            gem_y = g.col
            self.gems_list[index].move(
                self.screen_pad + gem_y * self.limit_distance_y, gem_x * self.limit_distance_x)
            self.gems_group.add(self.gems_list[index])

        pygame.draw.rect(
            self.viewer,
            (255, 255, 255), [self.screen_pad, 0, self.screen_width, self.screen_height])
        self.map_group.update()
        self.exits_group.update()
        self.gems_group.update()
        self.players_group.update()

        self.map_group.draw(self.viewer)
        self.exits_group.draw(self.viewer)
        self.gems_group.draw(self.viewer)
        self.players_group.draw(self.viewer)
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
