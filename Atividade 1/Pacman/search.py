# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this sho(5, 4uld return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
    
    
def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    """
    
    # Sempre começa no nó inicial
    current_pos      =  problem.getStartState()

    visited_nodes    =  []
    stack            =  util.Stack()

    # Esse cara aqui armazena em um dicionário qual nó é o pai de um determinado nó.
    # Por exemplo: parent[4,5] = (5,5) no layout tiny maze.
    parent           =  {}

    # Armazena quem são os vizinhos de um determinado nó N
    neighbor_nodes   =  []

    is_goal_found    =  False    
    neighbor_nodes   =  problem.getSuccessors(current_pos)

    # Marca o nó inicial como visitado
    visited_nodes.append(current_pos)

    # Como o nó inicial não tem pai, o nó pai dele é ele mesmo :P
    parent[ problem.getStartState() ] = problem.getStartState()

    # Populando a queue para simplificar o laço while abaixo
    for i in neighbor_nodes:
        stack.push(i)
        parent[ i[0] ] = current_pos
    
    while stack and not is_goal_found:
        current_pos = stack.pop()[0]
        visited_nodes.append(current_pos)
        
        # Se a posição atual for o objetivo
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        # Se não for, pegar os vizinhos do nó atual, adicionar os não-visitados
        # na queue e continuar o laço while.
        neighbor_nodes = problem.getSuccessors(current_pos)
        
        for i in neighbor_nodes:
            is_in_visited_nodes = False
            for j in visited_nodes:
                if i[0] == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                stack.push(i)
                parent[ i[0] ] = current_pos
    

    # Para saber o caminho, o melhor jeito é usar nosso dicionário que armazena o nó pai de outro nó.
    # Essa estratégia é boa no BFS porque pelo funcionamento do algoritmo, navegar pelos nó-pai vai sempre 
    # nos dar a menor distância entre dois nós. No DFS isso já não funcionaria, por exemplo.
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node]

    # Aqui eu começo no nó que é o final do labirinto e vou navegando de nó-filho para nó-pai até chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as direções e armazeno elas na variável solution
    while child_node != problem.getStartState():
        for neighbor, direction, cost in problem.getSuccessors(parent_node):
            if neighbor == child_node:
                solution.insert(0, direction)
                child_node  = parent_node
                parent_node = parent[parent_node]
                break

    return solution


def breadthFirstSearch(problem: SearchProblem):

    # Sempre começa no nó inicial
    current_pos      =  problem.getStartState()

    visited_nodes    =  []
    queue            =  util.Queue()

    # Esse cara aqui armazena em um dicionário qual nó é o pai de um determinado nó.
    # Por exemplo: parent[4,5] = (5,5) no layout tiny maze.
    parent           =  {}

    # Armazena quem são os vizinhos de um determinado nó N
    neighbor_nodes   =  []

    is_goal_found    =  False    
    neighbor_nodes   =  problem.getSuccessors(current_pos)

    # Marca o nó inicial como visitado
    visited_nodes.append(current_pos)

    # Como o nó inicial não tem pai, o nó pai dele é ele mesmo :P
    parent[ problem.getStartState() ] = problem.getStartState()

    # Populando a queue para simplificar o laço while abaixo
    for i in neighbor_nodes:
        queue.push(i)
        parent[ i[0] ] = current_pos
    
    while queue and not is_goal_found:
        current_pos = queue.pop()[0]
        visited_nodes.append(current_pos)
        
        # Se a posição atual for o objetivo
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        # Se não for, pegar os vizinhos do nó atual, adicionar os não-visitados
        # na queue e continuar o laço while.
        neighbor_nodes = problem.getSuccessors(current_pos)
        
        for i in neighbor_nodes:
            is_in_visited_nodes = False
            for j in visited_nodes:
                if i[0] == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                queue.push(i)
                parent[ i[0] ] = current_pos
    

    # Para saber o caminho, o melhor jeito é usar nosso dicionário que armazena o nó pai de outro nó.
    # Essa estratégia é boa no BFS porque pelo funcionamento do algoritmo, navegar pelos nó-pai vai sempre 
    # nos dar a menor distância entre dois nós. No DFS isso já não funcionaria, por exemplo.
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node]

    # Aqui eu começo no nó que é o final do labirinto e vou navegando de nó-filho para nó-pai até chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as direções e armazeno elas na variável solution
    while child_node != problem.getStartState():
        for neighbor, direction, cost in problem.getSuccessors(parent_node):
            if neighbor == child_node:
                solution.insert(0, direction)
                child_node  = parent_node
                parent_node = parent[parent_node]
                break

    return solution


def uniformCostSearch(problem: SearchProblem):
    
    current_pos    =  problem.getStartState()
    visited_nodes  =  []

    # Tem o mesmo propósito que o <parent> do BFS, mas aqui ele armazena um segundo valor: o gasto total para chegar
    # em um nó N a partir do nó inicial
    parent         =  {}

    # Items com prioridade maior ficam mais no final! Items com prioridade menos aparecem nas primeiras posições
    # (Priority, Number of Insertion, Item)
    p_queue        =  util.PriorityQueue()

    is_goal_found  =  False

    # Marca o nó inicial como visitado
    visited_nodes.append(current_pos)

    # Como o nó inicial não tem pai, o nó pai dele é ele mesmo, e obviamente, com custo zero
    parent[ problem.getStartState() ] = [ problem.getStartState(), 0 ]

    # Pega os nós vizinhos
    neighbor_nodes = problem.getSuccessors(current_pos)
    
    while not is_goal_found:
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        neighbor_nodes = problem.getSuccessors(current_pos)
        visited_nodes.append(current_pos)
        

        for position, direction, step_cost in neighbor_nodes:
            previous_cost = parent[current_pos][1]

            is_in_visited_nodes = False
            for j in visited_nodes:
                if position == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                # push(position, priority)
                # priority = step_cost do nó atual + custo total do caminho até o nó <position>
                p_queue.push(position, step_cost + previous_cost)
                parent[ position ] = [ current_pos, step_cost + previous_cost ]

        current_pos = p_queue.pop()

    # Para saber o caminho, o melhor jeito de novo é usar nosso dicionário que armazena o nó pai de outro nó.
    # Dessa forma, vamos seguir precisamente o caminho que o UCS acima percorreu pra achar o goalState
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node][0]    
    
    # Aqui eu começo no nó que é o final do labirinto e vou navegando de nó-filho para nó-pai até chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as direções e armazeno elas na variável solution
    while child_node != problem.getStartState():
        for neighbor, direction, cost in problem.getSuccessors(parent_node):
            if neighbor == child_node:
                solution.insert(0, direction)
                child_node  = parent_node
                parent_node = parent[parent_node][0]
                break
    
    return solution



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# TODO
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):

    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # O código do A* é literalmente a mesma coisa que o UCS, só que ele soma o valor da heurística junto ao step_cost
    # e ao custo acumulado do caminho na p_queue. Por isso eu só copiei e colei meu código do UCS, e quando eu faço um push
    # pra p_queue, eu somo junto o valor da heurística.

    current_pos    =  problem.getStartState()
    visited_nodes  =  []

    # Tem o mesmo propósito que o <parent> do BFS, mas aqui ele armazena um segundo valor: o gasto total para chegar
    # em um nó N a partir do nó inicial
    parent         =  {}

    # Items com prioridade maior ficam mais no final! Items com prioridade menos aparecem nas primeiras posições
    # (Priority, Number of Insertion, Item)
    p_queue        =  util.PriorityQueue()

    is_goal_found  =  False

    # Marca o nó inicial como visitado
    visited_nodes.append(current_pos)

    # Como o nó inicial não tem pai, o nó pai dele é ele mesmo, e obviamente, com custo zero
    parent[ problem.getStartState() ] = [ problem.getStartState(), 0 ]

    # Pega os nós vizinhos
    neighbor_nodes = problem.getSuccessors(current_pos)
    
    while not is_goal_found:
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        neighbor_nodes = problem.getSuccessors(current_pos)
        visited_nodes.append(current_pos)
        

        for position, direction, step_cost in neighbor_nodes:
            previous_cost = parent[current_pos][1]

            is_in_visited_nodes = False
            for j in visited_nodes:
                if position == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                # push(position, priority)
                # priority = step_cost do nó atual + custo total do caminho até o nó <position>
                p_queue.push(position, step_cost + previous_cost + heuristic(current_pos, problem))
                parent[ position ] = [ current_pos, step_cost + previous_cost + heuristic(current_pos, problem) ]

        current_pos = p_queue.pop()

    # Para saber o caminho, o melhor jeito de novo é usar nosso dicionário que armazena o nó pai de outro nó.
    # Dessa forma, vamos seguir precisamente o caminho que o UCS acima percorreu pra achar o goalState
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node][0]    
    
    # Aqui eu começo no nó que é o final do labirinto e vou navegando de nó-filho para nó-pai até chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as direções e armazeno elas na variável solution
    while child_node != problem.getStartState():
        for neighbor, direction, cost in problem.getSuccessors(parent_node):
            if neighbor == child_node:
                solution.insert(0, direction)
                child_node  = parent_node
                parent_node = parent[parent_node][0]
                break
    
    return solution

    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch