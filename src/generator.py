import random
import numpy  as np

def wilson_maze(width, height):
    """
    Generate a maze using Wilson's algorithm with fixed start and end points.

    :param width: Width of the maze grid (number of cells horizontally).
    :param height: Height of the maze grid (number of cells vertically).
    :return: A 2D NumPy array representing the maze, where 0 = wall, 1 = passage.
    """
    # Initialize maze grid with walls
    maze = np.zeros((height * 2 + 1, width * 2 + 1), dtype=np.uint8)

    # Fixed start and end points
    start = (0, 0)
    end = (width - 1, height - 1)

    # Initialize visited set with the start point
    visited = {start}

    # Directions: (dy, dx)
    directions = [ (-1, 0), (1, 0), (0, 1), (0, -1) ]

    def add_to_maze(path):
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            # Mark the current and next cells as passages
            maze[y1 * 2 + 1, x1 * 2 + 1] = 1
            maze[y2 * 2 + 1, x2 * 2 + 1] = 1

            # Mark the wall between cells as passage
            wall_x = x1 + x2 + 1
            wall_y = y1 + y2 + 1
            maze[wall_y, wall_x] = 1

    def remove_loops(path):
        no_loops_path = []
        seen = set()
        for cell in path:
            if cell in seen:
                # Remove the loop
                while no_loops_path[-1] != cell:
                    seen.remove(no_loops_path.pop())
            else:
                seen.add(cell)
                no_loops_path.append(cell)
        return no_loops_path
    
    # Generate the maze
    while len(visited) < width * height: # Until all cells are visited
        # Start a random walk from an unvisited cell
        current_cell = None
        while current_cell is None or current_cell in visited:
            current_cell = (random.randint(0, width - 1), random.randint(0, height - 1))

        path = [current_cell]
        # random walk
        while path[-1] not in visited:
            cx, cy = path[-1]
            direction = directions[random.randint(0, 3)]
            nx, ny = cx + direction[1], cy + direction[0]

            # Ensure the next cell is within bounds
            if 0 <= nx < width and 0 <= ny < height:
                path.append((nx, ny))

        # Erase loops in the random walk
        no_loops_path = remove_loops(path)

        # Add the processed path to the maze and the visited set
        visited.update(no_loops_path)
        add_to_maze(no_loops_path)

    # Ensure the start and end points are open
    # maze[start[1] * 2 + 1, start[0] * 2 + 1] = 2
    maze[end[1] * 2 + 1, end[0] * 2 + 1]     = 3

    return maze

def reset_maze_config(maze: np.ndarray) -> np.ndarray:
    """Reset maze while preserving the maze layout. Coins, Doors and Keys are removed.
    Args:
        maze (np.ndarray): original maze
    Returns:
        np.ndarray: cleaned maze
    """
    np.place(maze, maze != 0, 1)
    return maze