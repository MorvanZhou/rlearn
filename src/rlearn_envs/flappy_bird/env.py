import os
import random

import numpy as np
import pygame

from rlearn_envs.base import BaseEnv, Character, Background, Text

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
BIRD_IMAGE = os.path.join(FILE_DIR, "assets/bird.png")
PIPE_IMAGE = os.path.join(FILE_DIR, "assets/pipe.png")
SPEED = 12


class Bird(Character):
    def __init__(self, env: BaseEnv, images):
        super().__init__(
            images=images,
            scale=0.07,
        )
        self.env = env
        self.vy = 0
        self.g = 9.8
        self.dt = 0.3
        self.jump = False  # can be controlled by machine
        self.rect.inflate_ip(-10, -10)

    def update(self):
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE] or self.jump:
            self.vy = -30
            self.jump = False
        elif pressed[pygame.QUIT]:
            pygame.quit()
            exit()
        self.rect.y += self.vy * self.dt + 0.5 * self.g * self.dt ** 2
        self.vy += self.g * self.dt


class Pipe(Character):
    def __init__(self, env, images, colorkey):
        super().__init__(
            images=images,
            scale=0.3,
            colorkey=colorkey,
        )
        self.env = env

    def update(self):
        pass


class Env(BaseEnv):
    game_name = "Flappy Bird"
    size = (600, 350)

    def __init__(self, headless: bool = False):
        super().__init__(
            size=self.size,
            caption=self.game_name,
            mouse_visible=False,
            headless=headless,
        )
        self.bird = Bird(self, BIRD_IMAGE)
        self.pipe1 = Pipe(self, PIPE_IMAGE, colorkey=(0, 0, 0, 255))
        self.pipe2 = self.pipe1.copy()
        self.pipe2.image.set_colorkey((0, 0, 0, 255))
        self.pipe2.image, self.pipe2.rect = self.pipe2.flip(self.pipe2.image, False, True)

        self.pipe3 = self.pipe1.copy()
        self.pipe4 = self.pipe1.copy()
        self.pipe3.image.set_colorkey((0, 0, 0, 255))
        self.pipe4.image.set_colorkey((0, 0, 0, 255))
        self.pipe4.image, self.pipe4.rect = self.pipe4.flip(self.pipe4.image, False, True)

        self.background = Background(img_path=None, size=self.size)
        self.text = Text(font_name=None, size=25, color=(10, 10, 10), antialias=True, topleft=(10, 10))

        self.characters.add((self.bird, self.pipe1, self.pipe2, self.pipe3, self.pipe4))
        self.backgrounds.add_internal(self.background)
        self.texts.add(self.text)
        self.score = 0

    def step(self, action: np.ndarray = None):
        """

        :param action: 2D numpy array, [[0]] is doing nothing, [[1]] is pressing space key.
        :return: state, reward, done
        """
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, reward, True

        if self.pipe1.rect.right < 0:
            self.set_random_y(self.size[0], self.pipe1, self.pipe2)
            self.score += 1
            reward = 1

        if self.pipe3.rect.right < 0:
            self.set_random_y(self.size[0], self.pipe3, self.pipe4)
            self.score += 1
            reward = 1

        if action is not None and action[0] == 1:
            self.bird.jump = True
        self.bird.update()
        self.text.set_text("Score: {}".format(self.score))

        p_list = [self.pipe1, self.pipe2, self.pipe3, self.pipe4]
        for p in p_list:
            p.rect.move_ip(-SPEED, 0)
            if self.bird.rect.colliderect(p):
                reward = -1
                return None, reward, True
        if self.bird.rect.top > self.size[1] or self.bird.rect.top < 0:
            reward = -1
            return None, reward, True

        return None, reward, False

    def reset(self):
        self.set_random_y(self.size[0], self.pipe1, self.pipe2)
        self.set_random_y(self.size[0] * 1.5, self.pipe3, self.pipe4)
        self.bird.rect.center = (40, self.size[1] / 2)
        self.score = 0
        return None

    def set_random_y(self, x, p1, p2):
        y = random.randint(int(self.size[1] / 5), int(self.size[1] * 4 / 5))
        p1.rect.midtop = (x, y + 70)
        p2.rect.midbottom = (x, y - 70)
