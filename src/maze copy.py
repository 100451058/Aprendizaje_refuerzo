import enum
from optparse import Option
from re import M
from tkinter.tix import CELL
from typing import Literal, Optional

import copy

import heapq
import random
import numpy as np

import gymnasium as gym
from   gymnasium.spaces import Discrete, Dict, Box

from .generator import reset_maze_config, wilson_maze

import pygame
pygame.init()

# ----------------------------------------------------------------------------------------------
# Small Utils
# ----------------------------------------------------------------------------------------------

def match_coord(a: tuple[int, int], b: tuple[int, int]) -> bool: return a[0] == b[0] and a[1] == b[1]
def get_optimal_path(maze, start, end):
    start, end = tuple(start), tuple(end)

    possible_movement = [
        [-1,  0], # arriba
        [ 1,  0], # abajo
        [ 0, -1], # izquierda
        [ 0,  1]  # derecha
    ]

    def h(pos): return abs(end[0] - pos[0]) + abs(end[1] - pos[1])
    def is_valid(npos):
        y, x = npos
        border_check = 0 <= y < maze.shape[0] and 0 <= x < maze.shape[1]
        return border_check and maze[y, x] != 0

    openSet  = []
    closeSet = {}
    scores   = {tuple(start): 0}
    heapq.heappush(openSet, (0, start))

    def build_path(current, closeSet):
        path = []
        while current in closeSet:
            path.append(current)
            current = closeSet[current]
        path.append(start)
        path.reverse()
        return path
    
    while len(openSet) > 0: 
        cg, cpos = heapq.heappop(openSet)
        if cpos == end:
            return build_path(cpos, closeSet)

        for dirr in possible_movement:
            npos = (cpos[0] + dirr[0], cpos[1] + dirr[1])
            if not is_valid(npos): continue
            
            tentative = cg + 1
            if tuple(npos) not in scores or tentative < cg:
                scores[tuple(npos)] = tentative
                heapq.heappush(openSet, (tentative + h(npos), npos))
                closeSet[tuple(npos)] = cpos

    return []

# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class CellCodes(enum.IntEnum):
    WALL    = 0
    PASSAGE = 1
    END     = 3
    COIN    = 5
    PLAYER  = 7

CELL_SIZE: int = 10 # px

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, maze_size: tuple[int, int], key_task: bool = True, coin_task: Optional[int] = None, fixed_maze: bool = True, render_mode: Literal['human', 'rgb-array'] = 'human', render_surface: bool = False):
        """Maze Environment

        Args:
            maze_size (tuple[int, int]): shape of the maze (width, height). The final maze state is (2 * width + 1, 2 * height + 1)
            key_task (bool, optional): wheter keys and doors are added to the maze to stop movement. Defaults to True.
            coin_task (int, optional): Number of coins that are placed in the map. Defaults to None.
            fixed_maze (bool, optional): Wheter the maze is regenerated in each reset or it reuses the same maze. Coins and doors are reused too. Defaults to True
            render_mode (str, optional): Render model ('human' or 'rgb-array')
            render_surface (bool, optional): Determines if a surface or display are used for rendering. THis only works with `render_mode` equal to 'human'. Defaults to False
        """
        super().__init__()
        # define maze settings
        width, height = maze_size

        # the maze must contain the walls that are represented as an additional pixel.
        self.width, self.height = width, height
        self.maze_shape = (self.width * 2 + 1, self.height * 2 + 1) # (w, h)
        self.fixed_maze = fixed_maze

        # configure tasks
        self.task_key = key_task
        self.task_coin = 0 if coin_task is None else coin_task
        
        # define environment utils
        self.action_space = Discrete(4) # move right, bottom, left, top
        self.action2direction = { # x, y
            0: np.array([ 1,  0]), # right 
            1: np.array([ 0,  1]), # bottom
            2: np.array([-1,  0]), # left
            3: np.array([ 0, -1])  # top
        }
        self.coin_reward = int(100/coin_task) if coin_task > 0 else 0
        # the state of the maze is an image
        self.observation_space = Box(low = 0, high = 255, shape = (*self.maze_shape, 1), dtype = np.uint8)

        self._start = (1, 1) # 
        self._end   = list(np.array(self.maze_shape) - 2)
        self._curr  = self._start
        self._maze  = None # (h, w)
        self._coin_loc = []
        self._available_coins = []

        self.render_mode = render_mode
        if self.render_mode == 'human':

            size = self.maze_shape[0] * CELL_SIZE, self.maze_shape[1] * CELL_SIZE
            if render_surface:
                self.window = pygame.Surface(size)
            else:
                self.window = pygame.display.set_mode(size)
                pygame.display.set_caption("Maze Game")
            self.window_size = size

    def reset(self, random_start: bool = True) -> np.ndarray:
        """Reset environment to the default state

        Args:
            random_start (Optional[bool], optional): set the starting coordinate to a random position in the maze. Defaults to True.
        Returns:
            np.ndarray: new maze settings
        """
        if self._maze is None or not self.fixed_maze:
            self._maze = wilson_maze(self.width, self.height) # h, w
            self._maze = self.__place_coins(self._maze, self.task_coin)
        else:
            self._maze = reset_maze_config(self._maze)
            for cy, cx in self._coin_loc: self._maze[cy, cx] = int(CellCodes.COIN)
            self._maze[self._end[1], self._end[0]] = int(CellCodes.END)
            self._available_coins = copy.deepcopy(self._coin_loc)

        if random_start:
            posiciones_posibles = np.argwhere(self._maze == 1)
            eleccion = posiciones_posibles[random.randint(0, len(posiciones_posibles) - 1)]
            self._curr = (eleccion[0], eleccion[1]) # y, x
        else: self._curr = self._start

        maze = self._maze.copy()
        maze[self._curr[0], self._curr[1]] = int(CellCodes.PLAYER)
        return maze
    
    def __place_coins(self, maze: np.ndarray, number_coins: int) -> np.ndarray:
        w, h = self.maze_shape
        for _ in range(number_coins):
            while True:
                cx = random.randint(0, w)-1
                cy = random.randint(0, h)-1
                if maze[cy, cx] == CellCodes.PASSAGE:
                    self._coin_loc.append((cy, cx))
                    maze[cy, cx] = int(CellCodes.COIN)
                    break
        
        self._available_coins = copy.deepcopy(self._coin_loc)
        return maze


    def step(self, action: int, goal: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, float, bool]: 
        assert 0 <= action <= 3, "The action must be a number between 0 and 3. The mapping is (0: 'right', 1: 'bottom', 2: 'left', 3: 'right'). No " + str(action)

        # initial reward
        reward = 0

        # apply movement
        movement  = self.action2direction[action]

        # the state of the env is the full maze. not the movement or place of the player
        new_curr  = (self._curr[0] + movement[1], self._curr[1] + movement[0])
        if self._maze[new_curr[0], new_curr[1]] == 0: 
            return self._maze, -1, False, False, { "found_coin": -1 }
        self._curr = new_curr

        # set dummy goal if not enforced
        goal = goal or self._curr

        # mark player position
        new_state = self._maze.copy()
        new_state[self._curr[0], self._curr[1]] = int(CellCodes.PLAYER) 

        # update state
        # coin detection
        found_coin = -1
        if (self._maze[self._curr[0], self._curr[1]] == CellCodes.COIN) and match_coord(self._curr, goal): #Comprobamos que sea nuestro destino
            found_coin = list(filter(lambda c: match_coord(c[1], self._curr), enumerate(self._available_coins)))[0][0] 
            self._available_coins.pop(found_coin)
            self._maze[self._curr[0], self._curr[1]] = int(CellCodes.PASSAGE) # remove the coin
            reward = self.coin_reward
        
        # final state
        done   = match_coord(self._curr, self._end) and (len(self._available_coins) == 0) # no coins left
        reward = reward if not done else 100
        return new_state, reward, done, False, { "found_coin": found_coin }
    
    def get_current_position(self) -> tuple[int, int]: 
        """Player Position in the maze"""
        return self._curr 

    def get_coin_position(self) -> list[tuple[int, int]]: 
        """Coin Location in the maze"""
        return self._coin_loc 
    
    def render(self, mode: str = None):
        mode = mode or self.render_mode
        if mode == 'rgb-array':
            color_map = {
                0: (0, 0, 0),        # black
                1: (255, 255, 255),  # white
                2: (255, 0, 0),      # red
                3: (0, 128, 0),      # green
                4: (128, 0, 128),    # purple
                5: (255, 255, 0),    # yellow
                6: (255, 165, 0),    # orange
                7: (255, 0, 0)       # orange
            }
            maze_img = np.zeros((*self.maze_shape, 3), dtype=np.uint8)
            for i in range(self.maze_shape[0]): # w 
                for j in range(self.maze_shape[1]): # h
                    maze_img[j, i] = color_map[self._maze[j, i]]

            maze_img[self._curr[0], self._curr[1]] = color_map[7]
            return maze_img

        if mode != 'human': return None

        color_map = {
            0: 'black',
            1: 'white',
            2: 'red',
            3: 'green',
            4: 'purple',
            5: 'yellow',
            6: 'orange'
        }

        self.window.fill("black")
        for y in range(self.maze_shape[1]):
            for x in range(self.maze_shape[0]):
                color = color_map[self._maze[y, x]]
                rect  = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.window, color, rect, 0)

        current = (self._curr[1] * CELL_SIZE + CELL_SIZE // 2, self._curr[0] * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(self.window, "blue", current, CELL_SIZE // 2, 0)
        pygame.display.flip()

# hola > hello