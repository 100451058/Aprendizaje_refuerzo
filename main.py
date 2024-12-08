import pygame
import numpy as np
from random import randint
from src.maze import MazeEnv
from time import sleep

def main():
    env = MazeEnv((20, 20), False, 10, 'human')
    env.reset()
    while True:
        env.render()
        action = randint(0, 3)
        env.step(action)
        sleep(250 / 1000)

if __name__ == '__main__':
    main()

