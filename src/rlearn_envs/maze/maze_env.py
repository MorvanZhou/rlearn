import copy
import json
import math
import random
import time
import typing as tp

import numpy as np
import pygame

from rlearn import EnvWrapper
from rlearn_envs.maze import load


def game_over():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


class Maze(EnvWrapper):
    def __init__(self):
        super().__init__()
        self._map_data = self.load("./data/map.json")
        self.board = np.array(self._map_data.get("mapData", [[]]))
        self.copy_board = copy.deepcopy(self.board)
        self.row = self.board.shape[0]
        self.col = self.board.shape[1]
        # 获取地图网格信息
        if self.row == 0 or self.col == 0:
            raise ValueError("地图网格数据异常，请检查json文件中的maoData参数！")
        # 获取玩家信息
        self.players_info = self._map_data.get("players", [])
        self.players_num = len(self.players_info)
        if self.players_num == 0:
            raise ValueError("玩家数量不能少于1个，请检查json文件中的players参数！")
        # 获取终点信息
        self.exits_info = self._map_data.get("exits", [])
        if len(self.exits_info) != self.players_num:
            raise ValueError("终点数量和玩家数量不匹配，请检查json文件中的exits参数！")
        self.players_dict = dict()
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
        self.gem_type = ["red", "pink", "blue", "yellow", "purple"]
        self.gem_info = self._map_data.get("items", [])
        if len(self.gem_info) > 5:
            raise ValueError("items数量不能大于5个，请检查json文件中的items参数")
        self.gem_dict = dict()
        self.collections_config = self._map_data.get("collectionsConfig")
        self.effect_value = self.collections_config[0].get("effectValues", [10])[0]
        self.action_dict = {
            "u": [-1, 0],
            "d": [1, 0],
            "l": [0, -1],
            "r": [0, 1],
            "s": [0, 0]
        }
        self.finish = False
        self.info = None
        self.exits_pos_set = set()
        self.reward = 0

        # 用于render渲染
        pygame.init()
        self.FPS_CLOCK = pygame.time.Clock()
        self.screen_size = (1050, 1400)
        self.limit_distance_x = int(self.screen_size[0] / self.row)
        self.limit_distance_y = int(self.screen_size[0] / self.col)
        self.screen_size = (self.limit_distance_x * self.row, self.limit_distance_y * self.col + 350)
        self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
        # 加载地图元素
        self.bush = load.bush()
        self.grass = load.grass()
        self.land = load.land()
        self.stone = load.stone()
        self.tree = load.tree()
        self.water1 = load.water1()
        self.water2 = load.water2()
        self.wood1 = load.wood1()
        self.wood2 = load.wood2()
        # 加载玩家元素
        self.blue_player = load.blue_player()
        self.green_player = load.green_player()
        self.red_player = load.red_player()
        self.yellow_player = load.yellow_player()
        # 加载终点元素
        self.blue_exits = load.blue_exits()
        self.green_exits = load.green_exits()
        self.red_exits = load.red_exits()
        self.yellow_exits = load.yellow_exits()
        # 加载宝石图片
        self.pink_gem = load.pink_gem()
        self.red_gem = load.red_gem()
        self.blue_gem = load.blue_gem()
        self.yellow_gem = load.yellow_gem()
        self.purple_gem = load.purple_gem()
        self.bonus = load.bonus()

        self.map_param = [self.bush, self.grass, self.tree, self.stone,
                          self.water1, self.water2, self.wood1, self.wood2]
        self.player_param = [self.blue_player, self.green_player, self.yellow_player, self.red_player]
        self.player_queue = ["blue", "green", "yellow", "red"]
        self.exits_param = [self.blue_exits, self.green_exits, self.yellow_exits, self.red_exits]
        self.gem_param = {"red": self.red_gem,
                          "blue": self.blue_gem,
                          "yellow": self.yellow_gem,
                          "pink": self.pink_gem,
                          "purple": self.purple_gem}

        self.render_board = np.multiply(np.floor(np.random.rand(self.row, self.col) * len(self.map_param)), self.board)

    def reset(self):
        # 存储玩家id值，避免出现重复id
        exist_id = set()
        # 默认先手玩家为id = 0的玩家
        self.cur_player = 0
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
            exits_position = exits.get("position", None)
            if not exits_position:
                raise ValueError("终点初始坐标值不能为空，请检查json文件中的exits参数！")
            if exits_position["x"] < 0 or exits_position["x"] >= self.row or exits_position["y"] < 0 or \
                    exits_position["y"] >= self.col:
                raise ValueError("终点初始坐标值不在地图范围内，请检查json文件中的exits参数！")
            self.exits_pos_set.add((exits_position["x"], exits_position["y"]))
            self.players_dict[player_id] = dict()
            self.players_dict[player_id]["id"] = player_id
            self.players_dict[player_id]["position"] = {"x": player_position["x"], "y": player_position["y"]}
            self.players_dict[player_id]["action_point"] = player_action_point
            self.players_dict[player_id]["exits_position"] = exits_position
            self.players_dict[player_id]["score"] = 0
            self.players_dict[player_id]["bonus"] = 0
            self.players_dict[player_id]["gem"] = dict()
            self.copy_board[player_position["x"]][player_position["y"]] = 1
            self.copy_board[exits_position["x"]][exits_position["y"]] = 1
            for t in self.gem_type:
                self.players_dict[player_id]["gem"][t] = 0
        for gem_id in range(len(self.gem_info)):
            gem_info = self.gem_info[gem_id]
            gem_type = self.gem_type[gem_id]
            gem_position = gem_info.get("position", None)
            if not gem_position:
                raise ValueError("玩家初始坐标值不能为空，请检查json文件中的players参数！")
            if gem_position["x"] < 0 or gem_position["x"] >= self.row or gem_position["y"] < 0 or \
                    gem_position["y"] >= self.col:
                raise ValueError("宝石初始坐标值不在地图范围内，请检查json文件中的items参数！")
            self.gem_dict[gem_type] = {"x": gem_position["x"], "y": gem_position["y"]}
            self.copy_board[gem_position["x"]][gem_position["y"]] = 1
        self.finish = False
        self.info = None
        self.reward = 0
        return {"players_info": self.players_dict, "gem_info": self.gem_dict, "board": self.board}, self.reward, self.\
            finish

    def step(self, action):
        if self.finish:
            return {"players_info": self.players_dict, "gem_info": self.gem_dict, "board": self.board}, self.reward, \
                   self.finish
        player_id = self.cur_player % self.players_num
        player = self.players_dict[player_id]
        self.cur_player = player_id + 1
        move_data = self.action_dict[action]
        target_x = player["position"]["x"] + move_data[0]
        target_y = player["position"]["y"] + move_data[1]
        if action == "s":
            self.players_dict[player_id]["action_point"] -= self.map_weights["stay"]
        else:
            self.players_dict[player_id]["action_point"] -= self.map_weights["move"]
        if 0 <= target_x < self.row and 0 <= target_y < self.col and self.board[target_x][target_y] == 0:
            self.copy_board[player["position"]["x"]][player["position"]["y"]] = 0
            self.players_dict[player_id]["position"] = {"x": target_x, "y": target_y}
            self.copy_board[target_x][target_y] = 1
            for gem in self.gem_dict:
                gem_pos_x = self.gem_dict[gem]["x"]
                gem_pos_y = self.gem_dict[gem]["y"]
                if gem_pos_x == target_x and gem_pos_y == target_y:
                    self.players_dict[player_id]["score"] += self.effect_value
                    self.players_dict[player_id]["gem"][gem] += 1
                    while self.copy_board[gem_pos_x][gem_pos_y] != 0 or (gem_pos_x, gem_pos_y) in self.exits_pos_set:
                        gem_pos_x = math.floor(random.random() * self.row)
                        gem_pos_y = math.floor(random.random() * self.col)
                    self.copy_board[gem_pos_x][gem_pos_y] = 1
                    self.gem_dict[gem]["x"] = gem_pos_x
                    self.gem_dict[gem]["y"] = gem_pos_y
                    if len(list(
                            filter(lambda x:
                                   self.players_dict[player_id]["gem"][x] <=
                                   self.players_dict[player_id]["bonus"], self.players_dict[player_id]["gem"]))) == 0:
                        self.players_dict[player_id]["bonus"] += 1
                        self.players_dict[player_id]["score"] += 30
                    break
        if len(list(filter(lambda x: self.players_dict[x]["action_point"] > 0, self.players_dict))) == 0:
            self.finish = True
        return {"players_info": self.players_dict, "gem_info": self.gem_dict, "board": self.board}, self.reward, self.\
            finish

    def render(self):
        gem_width = 20
        font_size = 20
        pygame.display.set_caption("maze")
        self.viewer.fill(pygame.Color("white"))

        # 画直线
        for c in range(self.col):
            pygame.draw.lines(self.viewer, (255, 255, 255), True,
                              ((self.limit_distance_x * c, 0),
                               (self.limit_distance_x * c, self.col * self.limit_distance_x)), 1)
        for r in range(self.row):
            pygame.draw.lines(self.viewer, (255, 255, 255), True,
                              ((0, self.limit_distance_y * r),
                               (self.row * self.limit_distance_y, self.limit_distance_y * r)), 1)

        for x in range(self.row):
            for y in range(self.col):
                if self.board[x][y] == 0:
                    self.viewer.blit(
                        pygame.transform.scale(self.land, (self.limit_distance_y, self.limit_distance_x)),
                        (y * self.limit_distance_y, x * self.limit_distance_x))
                else:
                    self.viewer.blit(
                        pygame.transform.scale(
                            self.map_param[int(self.render_board[x][y])],
                            (self.limit_distance_y, self.limit_distance_x)),
                        (y * self.limit_distance_y, x * self.limit_distance_x))
        for index, player in enumerate(self.players_dict):
            exits_x = self.players_dict[player]["exits_position"]["x"]
            exits_y = self.players_dict[player]["exits_position"]["y"]
            self.viewer.blit(pygame.transform.scale(
                self.exits_param[index], (self.limit_distance_y, self.limit_distance_x)),
                (exits_y * self.limit_distance_y, exits_x * self.limit_distance_x))
            player_x = self.players_dict[player]["position"]["x"]
            player_y = self.players_dict[player]["position"]["y"]
            self.viewer.blit(pygame.transform.scale(
                self.player_param[index], (self.limit_distance_y, self.limit_distance_x)),
                (player_y * self.limit_distance_y, player_x * self.limit_distance_x))
            font = pygame.font.SysFont('inkfree', font_size)
            text = font.render(self.player_queue[index] + " id:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 20))
            text = font.render(str(index), True, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 20))
            text = font.render(self.player_queue[index] + " score:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 55))
            text = font.render(str(self.players_dict[player]["score"]), True, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 55))
            text = font.render(self.player_queue[index] + " step:", True, (0, 0, 0))
            self.viewer.blit(text, (50 + 250 * index, self.screen_size[0] + 90))
            text = font.render(str(self.players_dict[player]["action_point"]), False, (0, 0, 0))
            self.viewer.blit(text, (180 + 250 * index, self.screen_size[0] + 90))
            # 绘制玩家获取的宝石数量
            self.viewer.blit(pygame.transform.scale(self.pink_gem, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 125))
            text = font.render(" × " + str(self.players_dict[player]["gem"]["pink"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 122.5))
            self.viewer.blit(pygame.transform.scale(self.red_gem, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 160))
            text = font.render(" × " + str(self.players_dict[player]["gem"]["red"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 157.5))
            self.viewer.blit(pygame.transform.scale(self.yellow_gem, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 195))
            text = font.render(" × " + str(self.players_dict[player]["gem"]["yellow"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 192.5))
            self.viewer.blit(pygame.transform.scale(self.blue_gem, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 230))
            text = font.render(" × " + str(self.players_dict[player]["gem"]["blue"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 227.5))
            self.viewer.blit(pygame.transform.scale(self.purple_gem, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 265))
            text = font.render(" × " + str(self.players_dict[player]["gem"]["purple"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 262.5))
            self.viewer.blit(pygame.transform.scale(self.bonus, (gem_width, gem_width)),
                             (50 + 250 * index, self.screen_size[0] + 300))
            text = font.render(" × " + str(self.players_dict[player]["bonus"]), True, (0, 0, 0))
            self.viewer.blit(text, (70 + 250 * index, self.screen_size[0] + 297.5))
        for gem in self.gem_dict:
            gem_x = self.gem_dict[gem]["x"]
            gem_y = self.gem_dict[gem]["y"]
            self.viewer.blit(
                pygame.transform.scale(self.gem_param[gem], (self.limit_distance_y, self.limit_distance_x)),
                (gem_y * self.limit_distance_y, gem_x * self.limit_distance_x))
        pygame.display.update()
        game_over()
        time.sleep(0)  # 控制每帧渲染持续时间
        self.FPS_CLOCK.tick(20)  # 控制刷新速度，值越大刷新越快

    def load(self, map_json: tp.Any):
        with open(map_json) as file:
            map_data = json.load(file)
            return map_data


if __name__ == "__main__":
    maze = Maze()
    actions = ["u", "d", "l", "r", "s"]
    for _ in range(10):
        maze.reset()
        while True:
            state, reward, done = maze.step(actions[math.floor(random.random() * 5)])
            maze.render()
            if done:
                break
