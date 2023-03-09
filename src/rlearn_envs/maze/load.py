import os

import pygame

# 得到当前工程目录
img_dir = os.path.join(os.path.dirname(__file__), "assets", "img")
map_dir = os.path.join(img_dir, "map")
player_dir = os.path.join(img_dir, "penguin")
exits_dir = os.path.join(img_dir, "flag")
gem_dir = os.path.join(img_dir, "gem")
img_dict = dict(bush={"dir": os.path.join(map_dir, "bush.png"), "covert_alpha": False},
                grass={"dir": os.path.join(map_dir, "grass.png"), "covert_alpha": False},
                land={"dir": os.path.join(map_dir, "land.png"), "covert_alpha": False},
                stone={"dir": os.path.join(map_dir, "stone.png"), "covert_alpha": False},
                tree={"dir": os.path.join(map_dir, "tree.png"), "covert_alpha": False},
                water1={"dir": os.path.join(map_dir, "water1.png"), "covert_alpha": False},
                wood1={"dir": os.path.join(map_dir, "wood1.png"), "covert_alpha": False},
                water2={"dir": os.path.join(map_dir, "water2.png"), "covert_alpha": False},
                wood2={"dir": os.path.join(map_dir, "wood2.png"), "covert_alpha": False},
                blue_player={"dir": os.path.join(player_dir, "blue.png"), "covert_alpha": True},
                green_player={"dir": os.path.join(player_dir, "green.png"), "covert_alpha": True},
                red_player={"dir": os.path.join(player_dir, "red.png"), "covert_alpha": True},
                yellow_player={"dir": os.path.join(player_dir, "yellow.png"), "covert_alpha": True},
                blue_exits={"dir": os.path.join(exits_dir, "blue.png"), "covert_alpha": True},
                green_exits={"dir": os.path.join(exits_dir, "green.png"), "covert_alpha": True},
                red_exits={"dir": os.path.join(exits_dir, "red.png"), "covert_alpha": True},
                yellow_exits={"dir": os.path.join(exits_dir, "yellow.png"), "covert_alpha": True},
                blue_gem={"dir": os.path.join(gem_dir, "blue.png"), "covert_alpha": True},
                red_gem={"dir": os.path.join(gem_dir, "red.png"), "covert_alpha": True},
                yellow_gem={"dir": os.path.join(gem_dir, "yellow.png"), "covert_alpha": True},
                pink_gem={"dir": os.path.join(gem_dir, "pink.png"), "covert_alpha": True},
                purple_gem={"dir": os.path.join(gem_dir, "purple.png"), "covert_alpha": True},
                bonus={"dir": os.path.join(gem_dir, "bonus.png"), "covert_alpha": True},
                box={"dir": os.path.join(gem_dir, "box_close.png"), "covert_alpha": True})


def load_img(img_name):
    img_info = img_dict[img_name]
    if img_info["covert_alpha"]:
        return pygame.image.load(img_info["dir"]).convert_alpha()
    return pygame.image.load(img_info["dir"]).convert()
