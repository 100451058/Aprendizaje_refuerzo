import pygame
import numpy as np
from random import randint
from src.maze import MazeEnv
from time import sleep

def main():
    env = MazeEnv((32, 32), False, None, 'human')
    env.reset()
    while True:
        env.render()
        action = randint(0, 3)
        env.step(action)
        sleep(250 / 1000)

if __name__ == '__main__':
    main()

