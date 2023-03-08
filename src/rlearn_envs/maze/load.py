import os

import pygame

# 得到当前工程目录
img_dir = os.path.join(os.path.dirname(__file__), "assets", "img")

# 地图图片
map_img_dir = os.path.join(img_dir, "map")
bush_file = os.path.join(map_img_dir, "bush.png")
grass_file = os.path.join(map_img_dir, "grass.png")
land_file = os.path.join(map_img_dir, "land.png")
stone_file = os.path.join(map_img_dir, "stone.png")
tree_file = os.path.join(map_img_dir, "tree.png")
water1_file = os.path.join(map_img_dir, "water1.png")
water2_file = os.path.join(map_img_dir, "water2.png")
wood1_file = os.path.join(map_img_dir, "wood1.png")
wood2_file = os.path.join(map_img_dir, "wood2.png")

# 玩家图片
player_img_dir = os.path.join(img_dir, "penguin")
blue_player_file = os.path.join(player_img_dir, "blue.png")
green_player_file = os.path.join(player_img_dir, "green.png")
red_player_file = os.path.join(player_img_dir, "red.png")
yellow_player_file = os.path.join(player_img_dir, "yellow.png")

# 终点图片
flag_dir = os.path.join(img_dir, "flag")
blue_flag_file = os.path.join(flag_dir, "blue.png")
green_flag_file = os.path.join(flag_dir, "green.png")
red_flag_file = os.path.join(flag_dir, "red.png")
yellow_flag_file = os.path.join(flag_dir, "yellow.png")

# 宝石图片
gem_dir = os.path.join(img_dir, "gem")
blue_gem_file = os.path.join(gem_dir, "blue.png")
red_gem_file = os.path.join(gem_dir, "red.png")
yellow_gem_file = os.path.join(gem_dir, "yellow.png")
pink_gem_file = os.path.join(gem_dir, "pink.png")
purple_gem_file = os.path.join(gem_dir, "purple.png")
bonus_file = os.path.join(gem_dir, "bonus.png")
box_file = os.path.join(gem_dir, "box_close.png")


# 得到地图
def bush():
    bush = pygame.image.load(bush_file).convert()
    return bush


def box():
    box = pygame.image.load(box_file).convert()
    return box


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
