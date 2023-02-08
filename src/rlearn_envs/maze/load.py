import os

import pygame

# 得到当前工程目录
current_dir = os.path.split(os.path.realpath(__file__))[0]

# 地图图片
bush_file = current_dir + "/asserts/img/map/bush.png"
grass_file = current_dir + "/asserts/img/map/grass.png"
land_file = current_dir + "/asserts/img/map/land.png"
stone_file = current_dir + "/asserts/img/map/stone.png"
tree_file = current_dir + "/asserts/img/map/tree.png"
water1_file = current_dir + "/asserts/img/map/water1.png"
water2_file = current_dir + "/asserts/img/map/water2.png"
wood1_file = current_dir + "/asserts/img/map/wood1.png"
wood2_file = current_dir + "/asserts/img/map/wood2.png"

# 玩家图片
blue_player_file = current_dir + "/asserts/img/penguin/blue.png"
green_player_file = current_dir + "/asserts/img/penguin/green.png"
red_player_file = current_dir + "/asserts/img/penguin/red.png"
yellow_player_file = current_dir + "/asserts/img/penguin/yellow.png"

# 终点图片
blue_flag_file = current_dir + "/asserts/img/flag/blue.png"
green_flag_file = current_dir + "/asserts/img/flag/green.png"
red_flag_file = current_dir + "/asserts/img/flag/red.png"
yellow_flag_file = current_dir + "/asserts/img/flag/yellow.png"

# 宝石图片
blue_gem_file = current_dir + "/asserts/img/gem/blue.png"
red_gem_file = current_dir + "/asserts/img/gem/red.png"
yellow_gem_file = current_dir + "/asserts/img/gem/yellow.png"
pink_gem_file = current_dir + "/asserts/img/gem/pink.png"
purple_gem_file = current_dir + "/asserts/img/gem/purple.png"
bonus_file = current_dir + "/asserts/img/gem/bonus.png"


# 得到地图
def bush():
    bush = pygame.image.load(bush_file).convert()
    return bush


def grass():
    grass = pygame.image.load(grass_file).convert()
    return grass


def land():
    land = pygame.image.load(land_file).convert()
    return land


def stone():
    stone = pygame.image.load(stone_file).convert()
    return stone


def tree():
    tree = pygame.image.load(tree_file).convert()
    return tree


def water1():
    water1 = pygame.image.load(water1_file).convert()
    return water1


def water2():
    water2 = pygame.image.load(water2_file).convert()
    return water2


def wood1():
    wood1 = pygame.image.load(wood1_file).convert()
    return wood1


def wood2():
    wood2 = pygame.image.load(wood2_file).convert()
    return wood2


# 得到玩家
def blue_player():
    blue_player = pygame.image.load(blue_player_file).convert()
    return blue_player


def green_player():
    green_player = pygame.image.load(green_player_file).convert()
    return green_player


def red_player():
    red_player = pygame.image.load(red_player_file).convert()
    return red_player


def yellow_player():
    yellow_player = pygame.image.load(yellow_player_file).convert()
    return yellow_player


# 创建终点信息
def blue_exits():
    blue_exits = pygame.image.load(blue_flag_file).convert_alpha()
    return blue_exits


def green_exits():
    green_exits = pygame.image.load(green_flag_file).convert_alpha()
    return green_exits


def red_exits():
    red_exits = pygame.image.load(red_flag_file).convert_alpha()
    return red_exits


def yellow_exits():
    yellow_exits = pygame.image.load(yellow_flag_file).convert_alpha()
    return yellow_exits


# 宝石图片
def pink_gem():
    pink_gem = pygame.image.load(pink_gem_file).convert_alpha()
    return pink_gem


def red_gem():
    red_gem = pygame.image.load(red_gem_file).convert_alpha()
    return red_gem


def blue_gem():
    blue_gem = pygame.image.load(blue_gem_file).convert_alpha()
    return blue_gem


def yellow_gem():
    yellow_gem = pygame.image.load(yellow_gem_file).convert_alpha()
    return yellow_gem


def purple_gem():
    purple_gem = pygame.image.load(purple_gem_file).convert_alpha()
    return purple_gem


def bonus():
    bonus = pygame.image.load(bonus_file).convert_alpha()
    return bonus
