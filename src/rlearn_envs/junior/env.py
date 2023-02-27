import os
import random

import numpy as np
import pygame

from rlearn_envs.base import BaseEnv, Character, Background, Text

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
STAND_IMAGE = os.path.join(FILE_DIR, "assets/stand.png")
JUMP_IMAGE = os.path.join(FILE_DIR, "assets/jump.png")
GROUND_IMAGE = os.path.join(FILE_DIR, "assets/ground.png")
SPEED = 6


class Env(BaseEnv):
    game_name = "Junior"
    size = (600, 350)

    def __init__(self):
        super().__init__(
            size=self.size,
            caption=self.game_name,
            mouse_visible=False,
        )
        pygame.key.set_repeat(20, 150)

        self.jumper = Junior(self, JUMP_IMAGE, scale=0.05)
        self.stander = Junior(self, STAND_IMAGE, scale=0.1)

        self.ground1 = Ground(self, GROUND_IMAGE, scale=0.8)
        self.ground2 = self.ground1.copy()
        self.gy = self.size[1] - (self.ground1.rect.height + 10)

        self.background = Background(img_path=None, size=self.size)
        self.text_score = Text(font_name=None, size=25, color=(10, 10, 10), antialias=True, topleft=(10, 10))
        self.text_energy = Text(font_name=None, size=25, color=(10, 10, 10), antialias=True, topleft=(10, 30))

        self.characters.add((self.jumper, self.stander, self.ground1, self.ground2))
        self.backgrounds.add_internal(self.background)
        self.texts.add(self.text_score, self.text_energy)
        self.score = 0

        self.v = [0, 0]
        self.g = 13
        self.dt = 0.3
        self.energy = 0
        self.jumping = False
        self.bg_move = False
        self.buffer_x = 5

    def step(self, action: np.ndarray = None):
        """

        :param action: 2D numpy array, [[0]] is doing nothing, [[1]] is pressing space key.
        :return: state, reward, done
        """
        reward = 0
        vx = 35
        if action:
            assert isinstance(action, np.ndarray), TypeError
            self.energy = 100 * action[0]
            self.jumping = True
            self.text_energy.set_text(
                "Energy: {}".format(int(self.energy / 5) * "|" + int((100 - self.energy) / 5) * " " + "|"))
            self._set_jump_img(True)
            self.v = [vx, -self.energy * 1.3]
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, reward, True
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_SPACE, pygame.K_UP]:
                        self.energy = min(self.energy + 5, 100)
                        self.text_energy.set_text(
                            "Energy: {}".format(int(self.energy / 5) * "|" + int((100 - self.energy) / 5) * " " + "|"))
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_SPACE, pygame.K_UP]:
                        self.v = [vx, -self.energy * 1.3]
                        self.jumping = True
                        self._set_jump_img(True)

        while self.jumping:
            self._jump()
            if self.is_show:
                self.render_with_event_clear([self.jumper.rect, self.background.rect])

        while self.bg_move:
            self.text_score.set_text("Score: {}".format(self.score))
            self.text_energy.set_text("Energy: {}".format(int(100 / 5) * " " + "|"))
            self._move_to_origin()
            if not self.bg_move:
                reward = 1
            if self.is_show:
                self.render_with_event_clear([self.stander.rect, self.ground1.rect, self.ground2.rect])

        if self.jumper.rect.top > self.size[1]:
            return None, -1, True
        return None, reward, False

    def _move_to_origin(self):
        sl = self.stander.rect.left
        sr = self.stander.rect.right
        on_ground = self.ground1 if (sr > self.ground1.rect.left) and (sl < self.ground1.rect.right) else self.ground2
        other_ground = self.ground1 if on_ground == self.ground2 else self.ground2

        if other_ground.rect.right <= 0:
            other_ground.rect.left = random.randint(100, 300) + on_ground.rect.right

        dx = -20
        if on_ground.rect.left + dx < 0:
            dx = -on_ground.rect.left
            self.ground1.rect.move_ip(dx, 0)
            self.ground2.rect.move_ip(dx, 0)
            self.stander.rect.move_ip(dx, 0)
            self.jumper.rect.move_ip(dx, 0)
            self.bg_move = False
            return

        self.ground1.rect.move_ip(dx, 0)
        self.ground2.rect.move_ip(dx, 0)
        self.stander.rect.move_ip(dx, 0)
        self.jumper.rect.move_ip(dx, 0)

    def _set_jump_img(self, yes: bool):
        if yes:
            self.stander.set_alpha(0)
            self.jumper.set_alpha(1)
            self.jumper.rect.center = self.stander.rect.center
        else:
            self.stander.set_alpha(1)
            self.jumper.set_alpha(0)
            self.stander.rect.center = self.jumper.rect.center
            self.energy = 0

    def _jump(self):
        dy = self.v[1] * self.dt + 0.5 * self.g * self.dt ** 2
        dvy = self.g * self.dt

        # below ground
        if self.jumper.rect.bottom + dy > self.gy:
            if ((self.jumper.rect.right - self.buffer_x < self.right_ground.rect.left)
                and (self.jumper.rect.left + self.buffer_x > self.left_ground.rect.right)) \
                    or (self.jumper.rect.left + self.buffer_x > self.right_ground.rect.right):
                # in gap
                if self.jumper.rect.top > self.size[1]:
                    # to bottom
                    self.jumping = False
                    return
                self.jumper.rect.bottom += dy
                self.v[1] += dvy
                dx = self.v[0] * self.dt
                if (self.jumper.rect.right - self.buffer_x + dx > self.right_ground.rect.left) \
                        and (self.jumper.rect.right - self.buffer_x + dx < self.right_ground.rect.right):
                    self.jumper.rect.right = self.right_ground.rect.left - 1
                else:
                    self.jumper.rect.right += dx
                return
            else:
                # on ground
                self.jumper.rect.bottom = self.gy
                self.v = [0, 0]
                self.jumping = False
                if (self.jumper.rect.left + self.buffer_x <= self.right_ground.rect.right) \
                        and (self.jumper.rect.right - self.buffer_x >= self.right_ground.rect.left):
                    # on right
                    self.bg_move = True
                    self.score += 1
                else:
                    # on left
                    self.bg_move = False
                self._set_jump_img(False)
                return

        # in air
        self.jumper.rect.bottom += dy
        self.jumper.rect.left += self.v[0] * self.dt
        self.v[1] += dvy
        if self.jumper.rect.top < 0:
            self.jumper.rect.top = 0
        return

    @property
    def left_ground(self):
        return self.ground1 if self.ground1.rect.left < self.ground2.rect.left else self.ground2

    @property
    def right_ground(self):
        return self.ground1 if self.ground1.rect.left > self.ground2.rect.left else self.ground2

    def reset(self):
        self.ground1.rect.bottomleft = (0, self.size[1])
        self.ground2.rect.bottomleft = (self.ground1.rect.width + random.randint(100, 300), self.size[1])
        self.jumper.rect.bottomleft = [10, self.gy]
        self.score = 0
        self.energy = 0
        self.jumping = False
        self.bg_move = False
        self._set_jump_img(False)
        self.text_energy.set_text("Energy: {}".format(int(100 / 5) * " " + "|"))
        self.text_score.set_text("Score: 0")
        return None


class Junior(Character):
    def __init__(self, env: Env, images, scale=1., alpha=1., angle=0):
        super().__init__(
            images=images,
            scale=scale,
            alpha=alpha,
            angle=angle,
        )
        self.env = env

        self.rect.inflate_ip(-10, -10)

    def update(self):
        pass


class Ground(Character):
    def __init__(self, env: Env, images, scale=1., alpha=1., angle=0):
        super().__init__(
            images=images,
            scale=scale,
            alpha=alpha,
            angle=angle,
        )
        self.env = env

    def update(self):
        pass
