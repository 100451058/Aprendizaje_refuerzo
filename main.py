import pygame
import numpy as np
from random import randint
from src.maze import MazeEnv
from time import sleep

def main():
    env = MazeEnv((64, 64), True, 5, True, 'human', True)
    display = pygame.display.set_mode(env.window_size)
    env.reset(False)
    while True:
        env.render()
        action = randint(0, 3)
        env.step(action)
        display.blit(env.window, (0, 0))        
        sleep(250 / 1000)

if __name__ == '__main__':
    main()

