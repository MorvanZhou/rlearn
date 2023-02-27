from tkinter import PhotoImage

import pygame


class NoneSound:
    def play(self):
        pass


def load_image(path, colorkey=None):
    try:
        image = pygame.image.load(path)
    except pygame.error as e:
        print("Cannot load image:", path)
        raise SystemExit(e)
    image: pygame.Surface = image.convert()
    _set_colorkey(image, colorkey)
    return image


def _set_colorkey(image, colorkey):
    if colorkey is not None:
        if colorkey != -1:
            image.set_colorkey(colorkey)
    else:
        colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)


def load_gif(path, colorkey=None):
    frames = PhotoImage(path)
    images = []
    for frame in frames:
        image = pygame.image.frombuffer(frame, (frame.width, frame.hight), "RGB")
        image: pygame.Surface = image.convert()
        _set_colorkey(image, colorkey)
        images.append(image)
    return images


def load_sound(path):
    if not pygame.mixer:
        return NoneSound()
    try:
        sound = pygame.mixer.Sound(path)
    except pygame.error as e:
        print("Cannot load sound:", path)
        raise SystemExit(e)
    return sound
