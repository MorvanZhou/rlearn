import os
from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import pygame

from rlearn_envs.utils import load_image


class Character(pygame.sprite.Sprite):
    image: pygame.Surface
    rect: pygame.Rect
    images: List[pygame.Surface]
    _images: List[pygame.Surface]
    _images_path: List[str]

    def __init__(self, images, scale=1., alpha=1., angle=0, colorkey=None, frame_rate=3, frame_idx=0):
        super().__init__()
        if isinstance(images, str):
            self._images_path = [images]
        elif isinstance(images, list):
            assert len(images) > 0, ValueError
            assert isinstance(images[0], str), TypeError
            assert len(images) > frame_idx, ValueError
            self._images_path = images
        else:
            raise TypeError

        self._alpha = alpha
        self._scale = scale
        self._angle = angle
        self._frame_rate = frame_rate
        self._frame_idx = frame_idx
        self.images = []
        self._images = []

        for i, path in enumerate(self._images_path):
            _img = load_image(path, colorkey)
            self._images.append(_img)
            img, _ = self.set_scale(_img, scale)
            img, rect = self.set_angle(img, angle)
            self.images.append(img)
            if i == 0:
                self.rect = rect

        self.image = self.images[frame_idx]
        self.set_alpha(alpha)

    def update(self):
        raise NotImplemented

    def set_alpha(self, v):
        assert (v <= 1.) and (v >= 0), ValueError
        if self._alpha == v:
            return
        self._alpha = v
        alpha = int(255 * v)
        [img.set_alpha(alpha) for img in self.images]

    def set_scale(self, img: pygame.Surface, v):
        assert v > 0, ValueError
        self._scale = v
        w, h = int(img.get_width() * v), int(img.get_height() * v)
        center = img.get_rect().center
        _img: pygame.Surface = pygame.transform.scale(img, (w, h))
        _rect = _img.get_rect(center=center)
        return _img, _rect

    def set_scale_ip(self, v):
        assert v > 0, ValueError
        self._scale = v
        w, h = int(self._images[0].get_width() * v), int(self._images[0].get_height() * v)
        center = self._images[0].get_rect().center
        for i, _img in enumerate(self._images):
            self.images[i] = pygame.transform.scale(_img, (w, h))
        self.image = self.images[self._frame_idx]
        self.rect = self.image.get_rect(center=center)

    def set_angle(self, img: pygame.Surface, v):
        self._angle = v
        center = img.get_rect().center
        _img: pygame.Surface = pygame.transform.rotate(img, v)
        _rect = _img.get_rect(center=center)
        return _img, _rect

    def set_angle_ip(self, v):
        self._angle = v
        center = self._images[0].get_rect().center
        for i, _img in enumerate(self._images):
            self.images[i] = pygame.transform.rotate(_img, v)
        self.image = self.images[self._frame_idx]
        self.rect = self.image.get_rect(center=center)

    def set_rotozoom(self, img: pygame.Surface, angle, scale):
        self._angle = angle
        self._scale = scale
        center = img.get_rect().center
        _img = pygame.transform.rotozoom(img, angle, scale)
        _rect = _img.get_rect(center=center)
        return _img, _rect

    def set_rotozoom_ip(self, angle, scale):
        self._angle = angle
        self._scale = scale
        center = self._images[0].get_rect().center
        for i, _img in enumerate(self._images):
            self.images[i] = pygame.transform.rotozoom(_img, angle, scale)
        self.image = self.images[self._frame_idx]
        self.rect = self.image.get_rect(center=center)

    def set_frame_rate(self, v):
        assert v > 0, ValueError
        self._frame_rate = v

    def flip(self, img: pygame.Surface, x: bool, y: bool):
        center = img.get_rect().center
        _img = pygame.transform.flip(img, x, y)
        _rect = _img.get_rect(center=center)
        return _img, _rect

    def flip_ip(self, x: bool, y: bool):
        center = self._images[0].get_rect().center
        for i, _img in enumerate(self._images):
            self.images[i] = pygame.transform.flip(_img, x, y)
        self.image = self.images[self._frame_idx]
        self.rect = self.image.get_rect(center=center)

    def copy(self):
        a = Character(self._images_path, self.scale, self.alpha, self.angle)
        return a

    def next_frame(self):
        self._frame_idx += 1
        self._frame_idx = self._frame_idx % len(self.images)
        self.image = self.images[self._frame_idx]

    def set_frame(self, idx: int):
        self._frame_idx = idx
        self.image = self.images[self._frame_idx]

    def collide_boundary(self, env) -> (bool, bool, bool, bool):
        l, t, r, b = False, False, False, False
        if self.rect.midleft[0] < 0:
            l = True
        if self.rect.midtop[1] < 0:
            t = True
        if self.rect.midright[0] > env.size[0]:
            r = True
        if self.rect.midbottom[1] > env.size[1]:
            b = True

        return l, t, r, b

    @property
    def alpha(self):
        return self._alpha

    @property
    def scale(self):
        return self._scale

    @property
    def angle(self):
        return self._angle

    @property
    def frame_rate(self):
        return self._frame_rate


class Background(pygame.sprite.Sprite):
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self, img_path=None, size=None, fill=None, colorkey=None):
        super().__init__()
        self.path = img_path
        if img_path is None:
            assert type(size) in [tuple, list], TypeError
            assert len(size) == 2, ValueError
            self._image = pygame.Surface(size).convert()
            if not fill:
                self._image.fill((250, 250, 250, 1))
            else:
                self._image.fill(fill)
            self._rect = pygame.Rect(0, 0, *size)
        else:
            self._image: pygame.Surface = load_image(img_path, colorkey)
            self._rect = self._image.get_rect()
        self.image, self.rect = self._image, self._rect

    def update(self):
        raise NotImplemented


class Text:
    def __init__(self, font_name=None, size=23, color=(10, 10, 10), antialias=True, topleft=(0, 0)):
        if not pygame.font:
            self._font: pygame.font.Font = pygame.font.SysFont(font_name, size)
        else:
            self._font: pygame.font.Font = pygame.font.Font(font_name, size)
        self._color = color
        self._antialias = antialias
        self._text = ""
        self._topleft = topleft

    def draw(self, screen: pygame.Surface):
        t = self._font.render(self._text, self._antialias, self._color)
        screen.blit(t, t.get_rect(topleft=self._topleft))

    def set_text(self, text):
        self._text = text

    def set_topleft(self, x, y):
        self._topleft = (x, y)

    def set_color(self, c: Union[Tuple[int], List[int]]):
        self._color = c


class Texts:

    def __init__(self, *texts: Text):
        self._dict = {}
        if len(texts) != 0:
            self.add(*texts)

    def add(self, *texts: Text):
        for t in texts:
            if t in self._dict:
                raise KeyError
            self._dict[t] = True

    def remove(self, *texts: Text):
        for t in texts:
            if t in self._dict:
                self._dict.pop(t)

    def draw(self, screen: pygame.Surface):
        for t in self._dict.keys():
            t.draw(screen)


class BaseEnv(ABC):
    _screen: pygame.Surface
    backgrounds: pygame.sprite.Group
    _sprite_group: pygame.sprite.Sprite
    characters: pygame.sprite.Group
    texts: Texts
    render_rect_list: List

    size: Tuple
    game_name: str
    headless: bool
    _show: bool = True

    clock = pygame.time.Clock()
    default_clock_tick = 30

    def __init__(self, size: Union[List, Tuple], caption: str, mouse_visible: bool = False, headless=False):
        assert size is not None
        assert caption is not None

        self._caption = caption
        self._mouse_visible = mouse_visible
        self._headless = headless

        if headless:
            os.environ['SDL_VIDEODRIVER'] = "dummy"
        pygame.init()
        pygame.mixer.quit()
        pygame.display.set_caption(self._caption)
        pygame.mouse.set_visible(self._mouse_visible)
        self._screen = pygame.display.set_mode(size)

        self.backgrounds = pygame.sprite.Group()
        self.characters = pygame.sprite.Group()
        self.texts = Texts()
        self.render_rect_list = []

    @abstractmethod
    def step(self, action=None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def render(self):
        if not self._show:
            return
        self.backgrounds.draw(self._screen)
        self.characters.draw(self._screen)
        self.texts.draw(self._screen)
        if len(self.render_rect_list) > 0:
            pygame.display.update(self.render_rect_list)
        else:
            pygame.display.update()
        self.clock.tick(self.default_clock_tick)

    def render_with_event_clear(self, rect_list=None):
        if rect_list is not None:
            self.render_rect_list = rect_list
        self.render()
        self.render_rect_list.clear()
        pygame.event.clear()

    def set_show(self, yes: bool = True):
        self._show = yes

    @property
    def is_show(self):
        return self._show

    @property
    def name(self):
        return self._caption

    def close(self):
        pygame.display.quit()
        pygame.quit()
