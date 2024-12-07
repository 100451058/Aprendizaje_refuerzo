import pygame
import numpy as np
import random
from collections import deque

# Colores para PyGame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

CELL_SIZE = 40  # Tamaño de cada celda


class MazeEnv:
    def __init__(self, width=10, height=10, objects=3):
        self.width = width
        self.height = height
        self.num_objects = objects
        self.reset()

    def _generate_maze_kruskal(self):
        """
        Genera un laberinto utilizando el algoritmo de Kruskal.
        """
        maze = np.ones((self.height, self.width), dtype=int)  # Inicializar con paredes
        edges = []  # Lista de aristas entre celdas
        sets = {}  # Estructura de conjuntos disjuntos (Union-Find)

        # Inicializar las celdas como nodos individuales en conjuntos
        for x in range(0, self.height, 2):
            for y in range(0, self.width, 2):
                sets[(x, y)] = (x, y)
                # Crear aristas hacia las celdas vecinas
                if x + 2 < self.height:
                    edges.append(((x, y), (x + 2, y)))
                if y + 2 < self.width:
                    edges.append(((x, y), (x, y + 2)))

        # Funciones de Union-Find
        def find(cell):
            if sets[cell] != cell:
                sets[cell] = find(sets[cell])
            return sets[cell]

        def union(cell1, cell2):
            root1 = find(cell1)
            root2 = find(cell2)
            if root1 != root2:
                sets[root2] = root1

        # Algoritmo de Kruskal
        random.shuffle(edges)  # Barajar las aristas aleatoriamente
        for edge in edges:
            cell1, cell2 = edge
            if find(cell1) != find(cell2):  # Si están en diferentes conjuntos
                union(cell1, cell2)  # Unir conjuntos
                # Abrir camino entre las celdas
                wall_x = (cell1[0] + cell2[0]) // 2
                wall_y = (cell1[1] + cell2[1]) // 2
                maze[cell1] = 0
                maze[cell2] = 0
                maze[wall_x, wall_y] = 0

        return maze

    def _place_objects(self):
        """
        Coloca objetos en celdas transitables.
        """
        candidates = np.argwhere(self.maze == 0).tolist()
        objects = random.sample(candidates, self.num_objects)
        return objects

    def reset(self):
        """
        Reinicia el entorno y genera un nuevo laberinto.
        """
        self.maze = self._generate_maze_kruskal()
        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)
        self.objects = self._place_objects()
        self.player_position = self.start
        self.collected = [False] * self.num_objects

    def step(self, action):
        """
        Realiza una acción en el entorno.
        """
        x, y = self.player_position
        if action == 0:  # Arriba
            new_position = (x - 1, y) if x > 0 else self.player_position
        elif action == 1:  # Abajo
            new_position = (x + 1, y) if x < self.height - 1 else self.player_position
        elif action == 2:  # Izquierda
            new_position = (x, y - 1) if y > 0 else self.player_position
        elif action == 3:  # Derecha
            new_position = (x, y + 1) if y < self.width - 1 else self.player_position
        else:
            raise ValueError("Acción inválida")

        # Verificar si es transitable
        if self.maze[new_position] == 0:
            self.player_position = new_position

        # Recoger objetos
        for i, obj_pos in enumerate(self.objects):
            if self.player_position == obj_pos and not self.collected[i]:
                self.collected[i] = True

    def render(self, screen):
        """
        Renderiza el laberinto en PyGame.
        """
        screen.fill(BLACK)

        for x in range(self.height):
            for y in range(self.width):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.maze[x, y] == 1:  # Pared
                    pygame.draw.rect(screen, WHITE, rect)
                else:  # Espacio transitable
                    pygame.draw.rect(screen, BLACK, rect)
                    pygame.draw.rect(screen, BLUE, rect, 1)

        pygame.draw.rect(screen, GREEN, (self.start[1] * CELL_SIZE, self.start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, (self.end[1] * CELL_SIZE, self.end[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for i, obj in enumerate(self.objects):
            if not self.collected[i]:
                pygame.draw.circle(screen, YELLOW, 
                                   (obj[1] * CELL_SIZE + CELL_SIZE // 2, obj[0] * CELL_SIZE + CELL_SIZE // 2), 
                                   CELL_SIZE // 4)

        pygame.draw.circle(screen, BLUE, 
                           (self.player_position[1] * CELL_SIZE + CELL_SIZE // 2, 
                            self.player_position[0] * CELL_SIZE + CELL_SIZE // 2), 
                           CELL_SIZE // 3)


# Prueba del entorno con visualización PyGame
def main():
    pygame.init()
    width, height = 21, 21  # Tamaño del laberinto
    env = MazeEnv(width=width, height=height, objects=3)
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Maze - Kruskal's Algorithm")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            env.step(0)  # Arriba
        elif keys[pygame.K_DOWN]:
            env.step(1)  # Abajo
        elif keys[pygame.K_LEFT]:
            env.step(2)  # Izquierda
        elif keys[pygame.K_RIGHT]:
            env.step(3)  # Derecha

        env.render(screen)
        pygame.display.flip()
        clock.tick(10)  # 10 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
