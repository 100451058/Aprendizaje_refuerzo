from typing import Optional

import heapq
import random
import numpy as np

import gymnasium as gym
from   gymnasium.spaces import Discrete, Dict, Box

from .maze_generator import wilson_maze


def get_optimal_path(maze, start, end):
    start, end = tuple(start), tuple(end)

    possible_movement = [
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1]
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
            print("Path Found")
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


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, maze_size: tuple[int, int], key_task: bool = True, coin_task: Optional[int] = None):
        """Maze Environment

        Args:
            maze_size (tuple[int, int]): shape of the maze (width, height). The final maze state is (2 * width + 1, 2 * height + 1)
            key_task (bool, optional): wheter keys and doors are added to the maze to stop movement. Defaults to True.
            coin_task (int, optional): Number of coins that are placed in the map. Defaults to None.
        """
        super().__init__()
        
        # define maze settings
        width, height = maze_size

        # the maze must contain the walls that are represented as an additional pixel.
        self.width, self.height = width, height
        self.maze_shape = (self.width * 2 + 1, self.height * 2 + 1)

        # configure tasks
        self.task_key = key_task
        self.task_coin = coin_task
        
        # define environment utils
        self.action_space = Discrete(4) # move right, bottom, left, top
        self.action2direction = {
            0: np.array([ 0,  1]), # right 
            1: np.array([ 1,  0]), # bottom
            2: np.array([ 0, -1]), # left
            3: np.array([-1,  0]) # top
        }
        # the state of the maze is an image
        self.observation_space = Box(low = 0, max = 255, shape = (*self.maze_shape, 1), dtype = np.uint8)

        self._start = (1, 1)
        self._end   = tuple(np.array(self.maze_shape) - 2)
        self._curr  = self._start

    def reset(self) -> np.ndarray:
        np.random.seed(1001)
        random.seed(1001)

        self._maze = wilson_maze(self.width, self.height)
        self._curr = self._start
    
    def step(self, action: int ): 
        assert 0 <= action <= 3, "The action must be a number between 0 and 4. The mapping is (0: 'right', 1: 'bottom', 2: 'left', 3: 'right')"
        
        # apply movement
        movement = self.action2direction[action]
        self._curr = (self._curr[0] + movement[0], self._curr[1] + movement[1])
        
        # update state
        # if keys are removed
        new_state = self._maze.copy()
        new_state[self._curr[0], self._curr[1]] = 5
        
        done   = self._curr == self._end
        reward = 0 if not done else 100
        return new_state, reward, done
