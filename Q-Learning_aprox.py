import numpy as np
from src.maze import MazeEnv
import pygame
from tqdm import tqdm
import time

# Parámetros
episodios_usuario = 3
episodios_agente = 1000
episodios_test = 10

#Parametros estadisticos
eficiencia = []
pasos_por_episodio = []
recompensa_por_episodio = []
valor_politica_optima = []

# class Manager:
#     def __init__(self, env):
#         self.env: MazeEnv = env
#         self.q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], env.maze_shape[0], env.maze_shape[1]))  # Q-table para el manager
#         self.alpha = 0.1
#         self.gamma = 0.9
#         self.epsilon = 0.1
#         self.end = env._end
#         self.learned_paths = []

#     def select_goal(self, state):
#         coins = np.argwhere(self.env._maze == 5).tolist()
#         if np.random.rand() < self.epsilon:
#             if len(coins) > 0:
#                 return coins[np.random.randint(len(coins))]
#             else:
#                 return self.env._end
#         else:
#             if len(coins)>0:
#                 if np.random.rand() < self.epsilon*2:
#                     distances = [sum(abs(np.array(state) - np.array(coin))) for coin in coins]
#                     return coins[np.argmin(distances)]
#                 else:
#                     q_values = [self.q_table[state[0], state[1], coin[0], coin[1]] for coin in coins]
#                     return coins[np.argmax(q_values)]
#             else:
#                 return self.end

#     def update_q_table(self, state, goal, reward, next_state,steps):
#         # Penalizar la ganancia dependiendo del tiempo sin llegar a un goal
#         time_penalty = 0.9**steps
#         reward = reward * time_penalty
#         #Actualizar la tabla Q
#         end = [int(item) for item in self.end]
#         best_next_q = np.max([self.q_table[next_state[0], next_state[1], g[0], g[1]] for g in np.argwhere(self.env._maze == 5).tolist() + [end]])
#         self.q_table[state[0], state[1], goal[0], goal[1]] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state[0], state[1], goal[0], goal[1]])
        
class Manager:
    def __init__(self, env):
        self.env: MazeEnv = env
        self.q_table = np.ones((env.maze_shape[0], env.maze_shape[1], env.maze_shape[0], env.maze_shape[1]))  # Q-table para el manager
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.end = env._end
        self.learned_paths = []
        self.prev_goal = None

    def select_goal(self, state):
        coins = np.argwhere(self.env._maze == 5).tolist()
        self.prev_goal = state
        if np.random.rand() < self.epsilon:
            if len(coins) > 0:
                return coins[np.random.randint(len(coins))]
            else:
                return self.env._end
        else:
            if len(coins)>0:
                #Elige la mejor opción
                q_values = [self.q_table[state[0], state[1], coin[0], coin[1]] for coin in coins]
                return coins[np.argmax(q_values)]
            else:
                return self.end
        

    def update_q_table(self, state, reward,steps):
        # Penalizar la ganancia dependiendo del tiempo sin llegar a un goal
        time_penalty = 0.9**steps
        reward = reward * time_penalty
        #Actualizar la tabla Q
        self.q_table[self.prev_goal[0], self.prev_goal[1], state[0], state[1]] = reward



class Worker:
    def __init__(self, env):
        self.env = env
        self.q_tables =  {} # Q-table para el worker
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.learned_paths = []

    def select_action(self, state, goal):
        self.epsilon = self.epsilon *0.95 if self.epsilon > 0.10 else 0.1
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_table = self.q_tables[tuple(goal)]
            return np.argmax(q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state, goal,steps):
        # Penalizar la ganancia dependiendo del tiempo sin llegar a un goal
        time_penalty = 0.95**steps
        reward = reward * time_penalty

        q_table = self.q_tables[tuple(goal)]
        best_next_q = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] += self.alpha * (reward + self.gamma * best_next_q - q_table[state[0], state[1], action])
        self.q_tables[tuple(goal)] = q_table  # Guardar el q_table actualizado
    
    def transfer_learning(self, goal):
        # print(f"Transfer learning to goal: {goal}")
        if(self.learned_paths == []):
            self.learned_paths.append(goal)
            q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], 4))
            self.q_tables[tuple(goal)] = q_table
        else:
            if goal in self.learned_paths: #APRENDIDO
                self.q_table = self.q_tables[tuple(goal)]
                self.epsilon = 0.1
            else: #NO APRENDIDO
                # Calcular path mas parecido
                min_distance = -1
                min_goal = None
                for learned_goal in self.learned_paths:
                    distance = sum(abs(np.array(goal)-np.array(learned_goal)))
                    # distance = np.linalg.norm(np.array(goal) - np.array(learned_goal))
                    if distance < min_distance or min_distance == -1:
                        min_distance = distance
                        min_goal = learned_goal
                
                #Crear nueva política
                if(min_distance < 3): #Si tengo una política parecida
                    self.q_tables[tuple(goal)] = self.q_tables[tuple(min_goal)]
                    # Le doy más libertad a investigar
                    self.epsilon = 0.5
                    
                else: #Si no tengo una política parecida
                    q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], 4))
                    self.q_tables[tuple(goal)] = q_table
                    self.epsilon = 0.1

                # Añadir nuevo objetivo
                self.learned_paths.append(goal)

# Función para mapear las teclas a acciones
def get_action_from_key(key):
    if key == pygame.K_RIGHT:
        return 0
    elif key == pygame.K_DOWN:
        return 1
    elif key == pygame.K_LEFT:
        return 2
    elif key == pygame.K_UP:
        return 3
    return None

print("mazed")

env = MazeEnv((10, 10), False, 3, 'human')
manager = Manager(env)
worker = Worker(env)

# ITERACION DEL USUARIO
print("AYUDANDO AL AGENTE...")
# Bucle principal para controlar el agente con el teclado
for i in range(episodios_usuario):
    _ = env.reset()
    state = env.get_current_position()
    done  = False
    goal  = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    worker.transfer_learning(goal)
    print(f"user episode: {i+1:02d}/{episodios_usuario:02d} -> goal: {goal}")
    movimientos = 0
    env.render()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
            elif event.type == pygame.KEYDOWN:
                action = get_action_from_key(event.key)
                if action is not None:
                    _, reward, done = env.step(action,goal)
                    next_state = env.get_current_position()
                    reward = 1 if reward<1 else reward
                    worker.update_q_table(state, action, reward, next_state,goal,0)
                    state = next_state

                    movimientos+=1

                    # Si se alcanza el objetivo, el episodio termina
                    if np.array_equal(state, goal):
                        goal = manager.select_goal(state)
                        manager.update_q_table(state, reward,0)
                        worker.transfer_learning(goal)
                        print(f"user episode: {i+1:02d}/{episodios_usuario:02d} -> goal: {goal}")
        
        env.render()
    print()
env.close()
                    

# ITERACION DEL AGENTE
print("EL AGENTE SOLO...")
# Iteración por episodios
# env._place_coins(env._maze, 3)
for i in tqdm(range(episodios_agente)):
    _ = env.reset()
    state = env.get_current_position()
    done = False
    goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    worker.transfer_learning(goal)
    manager.gamma = manager.gamma*0.99 if manager.gamma > 0.4 else 0.4
    
    #Variables estadisticas
    movimientos = 0
    eginak = 0
    reward_acumulado = 0

    # Iteración por pasos
    while not done and movimientos < 500:
        action = worker.select_action(state, goal)
        _, reward, done = env.step(action,goal)
        reward_acumulado += reward
        next_state = env.get_current_position()
        
        worker.update_q_table(state, action, reward, next_state,goal,movimientos)
        state = next_state
        env.render()
        movimientos+=1
        # Si se alcanza el objetivo, selecciona un nuevo objetivo
        if np.array_equal(state, goal):
            goal = manager.select_goal(state)
            manager.update_q_table(state, reward,movimientos)
            worker.transfer_learning(goal)
            eginak+=1
        
        if(done):
            eginak+=1
    
    pasos_por_episodio.append(movimientos)
    eficiencia.append(eginak)
    recompensa_por_episodio.append(reward_acumulado)

    


    time.sleep(0.01)

env.close()

# Resultados del aprendizaje
print("Porcentaje de episodios completados: ", eginak/episodios_agente)

logros = 0
for i in range(episodios_test):
    #Probar la eficiencia de la politica
    _ = env.reset()
    state = env.get_current_position()
    done = False
    goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    movimientos = 0
    while not done and movimientos < 300:
        action = worker.select_action(state, goal)
        _, reward, done = env.step(action,goal)
        next_state = env.get_current_position()
        
        state = next_state
        env.render()

        movimientos+=1

        # Si se alcanza el objetivo, selecciona un nuevo objetivo
        if np.array_equal(state, goal):
            goal = manager.select_goal(state)
        
        if(done):
            print("SE LOGRO ENCONTRAR EL CAMINO OPTIMO")
            logros+=1

print("Porcentaje de eficiencia de la política: ", 100*(logros/episodios_test))



# Estadisticas
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(pasos_por_episodio)
plt.title("Pasos por episodio ("+str(sum(pasos_por_episodio)/episodios_agente)+")")
plt.subplot(1,3,2)
plt.plot(eficiencia)
plt.title("Eficiencia ("+str(sum(eficiencia)/episodios_agente)+")")
plt.subplot(1,3,3)
plt.plot(recompensa_por_episodio)
plt.title("Recompensa por episodio ("+str(sum(recompensa_por_episodio)/episodios_agente)+")")
plt.show()
