import os
import random

import numpy as np
import pygame

from rlearn_envs.base import BaseEnv, Character, Background, Text

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DINO_RUN1_IMAGE = os.path.join(FILE_DIR, "assets/dino_run1.png")
DINO_RUN2_IMAGE = os.path.join(FILE_DIR, "assets/dino_run2.png")
DINO_DOWN1_IMAGE = os.path.join(FILE_DIR, "assets/dino_down1.png")
DINO_DOWN2_IMAGE = os.path.join(FILE_DIR, "assets/dino_down2.png")
DINO_COLLI_IMAGE = os.path.join(FILE_DIR, "assets/dino_collision.png")
DINO_STAND_IMAGE = os.path.join(FILE_DIR, "assets/dino_stand.png")
PLANT_IMAGE = os.path.join(FILE_DIR, "assets/plant.png")
GROUND_IMAGE = os.path.join(FILE_DIR, "assets/ground.png")
BIRD1_IMAGE = os.path.join(FILE_DIR, "assets/bird1.png")
BIRD2_IMAGE = os.path.join(FILE_DIR, "assets/bird2.png")
SPEED = 6


class Env(BaseEnv):
    game_name = "Jumping dino"
    size = (600, 250)

    def __init__(self):
        super().__init__(
            size=self.size,
            caption=self.game_name,
            mouse_visible=False,
        )
        pygame.key.set_repeat(20, 200)

        self.dino_colli = Dino(self, DINO_COLLI_IMAGE, scale=0.05)
        self.dino_run = Dino(self, [DINO_RUN1_IMAGE, DINO_RUN2_IMAGE], scale=0.05)
        self.dino_down = Dino(self, [DINO_DOWN1_IMAGE, DINO_DOWN2_IMAGE], scale=0.15)
        self.dino_stand = Dino(self, DINO_STAND_IMAGE, scale=0.35)
        self.plant1 = Plant(self, PLANT_IMAGE, scale=0.05)
        self.plant2 = Plant(self, PLANT_IMAGE, scale=0.05)
        self.ground1 = Ground(self, GROUND_IMAGE, scale=1)
        self.ground2 = Ground(self, GROUND_IMAGE, scale=1)
        self.bird = Bird(self, [BIRD1_IMAGE, BIRD2_IMAGE], scale=0.2)

        self.background = Background(img_path=None, size=self.size)
        self.text_score = Text(font_name=None, size=25, color=(10, 10, 10), antialias=True, topleft=(10, 10))

        self.characters.add(
            self.dino_colli, self.dino_run, self.dino_stand,
            self.dino_down, self.plant1, self.plant2,
            self.ground1, self.ground2, self.bird)
        self.backgrounds.add_internal(self.background)
        self.texts.add(self.text_score)
        self.score = 0
        self.h_score = 0
        self.gh = self.size[1] - 50
        self.dino_y = self.gh
        self.dino_energy = 0
        self.t = 0
        self.g = 13
        self.dt = 0.3
        self.v = -40
        self.dx = -8
        self.down = False
        self.accelerate = False

    def step(self, action: np.ndarray = None):
        """

        :param action: 2D numpy array, [[0]] is doing nothing, [[1]] is pressing space key.
        :return: state, reward, done
        """
        reward = 0
        if action:
            assert isinstance(action, np.ndarray), TypeError
            if action[0] > 0:
                self.dino_energy += 1
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.dino_energy += 1
                    elif event.key == pygame.K_DOWN:
                        if self.dino_energy == 0:
                            self.down = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        pass
                    elif event.key == pygame.K_DOWN:
                        if self.dino_energy == 0:
                            self.down = False

        self.t += 1
        if self.t > 200:
            self.t -= 200
            self.dx -= 1
            self.dx = max(self.dx, -30)

        if self.dino_energy > 0:
            self._jump()
        else:
            self._run()

        self.ground1.update()
        self.ground2.update()
        self._update_obstacle()
        collide = self._check_collision()
        if collide:
            self._collide()
            self.h_score = self.score
            return None, -1, True
        self.text_score.set_text("Highest: {} | Score: {}".format(self.h_score, self.score))
        return None, reward, False

    def _update_obstacle(self):
        not_on_queen = []
        most_right = self.plant1
        for item in [self.plant1, self.plant2, self.bird]:
            if item.rect.right <= 0:
                not_on_queen.append(item)
            if most_right.rect.right < item.rect.right:
                most_right = item
            item.rect.left += self.dx

        if self.t % self.bird.frame_rate == 0:
            self.bird.next_frame()

        if len(not_on_queen) == 0:
            return

        for item in not_on_queen:
            if isinstance(item, Plant):
                item.rect.left = most_right.rect.right + random.randint(300, 600)
                break
            else:
                if self.dx > -11:
                    continue
                if item.rect.right < 0:
                    x = most_right.rect.right + random.randint(300, 600)
                    y = self.gh - 30 if random.randint(0, 1) == 0 else self.gh
                    self.bird.rect.bottomright = (x, y)
                break

    def _check_collision(self):
        for p in [self.plant1, self.plant2]:
            if p.rect.collidepoint(self.dino_run.rect.left, self.dino_y):
                return True
            if p.rect.collidepoint(self.dino_run.rect.right, self.dino_y):
                return True
        if self.down and self.bird.rect.colliderect(self.dino_down.rect):
            return True
        elif (not self.down) and self.bird.rect.colliderect(self.dino_stand.rect):
            return True
        return False

    def _jump(self):
        if self.dino_energy >= 2 and not self.accelerate:
            self.accelerate = True
            self.v -= 15
        self.dino_stand.set_alpha(1)
        self.dino_run.set_alpha(0)
        self.dino_down.set_alpha(0)
        self.dino_colli.set_alpha(0)
        dy = self.v * self.dt + 0.5 * self.g * self.dt ** 2
        dv = self.g * self.dt
        self.dino_y += dy
        self.v += dv

        if self.dino_y > self.gh:
            self.dino_energy = 0
            self.dino_y = self.gh
            self.v = -40
            self.accelerate = False
            self.score += 1

        self.dino_stand.rect.bottom = self.dino_y

    def _run(self):
        if self.down:
            self.dino_down.update()
            self.dino_run.set_alpha(0)
            self.dino_down.set_alpha(1)
        else:
            self.dino_run.update()
            self.dino_down.set_alpha(0)
            self.dino_run.set_alpha(1)
        if self.dino_stand.alpha == 1 or self.dino_colli.alpha == 1:
            self.dino_stand.set_alpha(0)
            self.dino_colli.set_alpha(0)

    def _collide(self):
        if self.is_show:
            self.dino_stand.set_alpha(0)
            self.dino_run.set_alpha(0)
            self.dino_down.set_alpha(0)
            self.dino_colli.set_alpha(1)
            for _ in range(20):
                self.render_with_event_clear([self.dino_run.rect, self.dino_stand, self.dino_colli])

    def reset(self):
        self.ground1.rect.topleft = (0, self.gh - 2)
        self.ground2.rect.topleft = (self.ground1.rect.w, self.gh - 2)
        self.dino_colli.rect.bottomleft = (50, self.gh)
        self.dino_run.rect.bottomleft = (50, self.gh)
        self.dino_stand.rect.bottomleft = (50, self.gh)
        self.plant1.rect.bottomleft = (self.size[0], self.gh + 15)
        self.plant2.rect.bottomleft = (self.size[0] * 1.5, self.gh + 15)
        self.bird.rect.bottomright = (-10, self.gh - 30)
        self.dino_down.rect.bottomleft = (50, self.gh)
        self.score = 0
        self.dx = -8
        self.t = 0
        self.down = False
        self.text_score.set_text("Highest: {} | Score: 0".format(self.h_score))
        self.dino_run.set_alpha(1)
        self.dino_down.set_alpha(0)
        self.dino_colli.set_alpha(0)
        self.bird.set_alpha(1)
        return None


class Dino(Character):
    def __init__(self, env: Env, images, scale=1., alpha=1., angle=0, frame_rate=3):
        super().__init__(
            images=images,
            scale=scale,
            alpha=alpha,
            angle=angle,
            frame_rate=frame_rate,
        )
        self.env = env
        self.rect.inflate_ip(-30, -10)

    def update(self):
        if self.env.t % self.frame_rate == 0:
            self.next_frame()


class Plant(Character):
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


class Bird(Character):
    def __init__(self, env: Env, images, scale=1., alpha=1., angle=0):
        super().__init__(
            images=images,
            scale=scale,
            alpha=alpha,
            angle=angle,
        )
        self.env = env

    def update(self):
        if self.env.t % self.frame_rate == 0:
            self.next_frame()


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
        if self.rect.right < 0:
            self.rect.left = self.rect.right + self.rect.width
        self.rect.left += self.env.dx
