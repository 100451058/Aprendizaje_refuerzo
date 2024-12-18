import numpy as np
import pygame

from src.maze import MazeEnv
from src.maze import CellCodes, CELL_SIZE

def main():
    base_size = 8, 8
    env = MazeEnv(base_size, False, 0, True, 'human', None)
    

if __name__ == '__main__': main()