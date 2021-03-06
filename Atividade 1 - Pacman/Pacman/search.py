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
    
    # Sempre come??a no n?? inicial
    current_pos      =  problem.getStartState()

    visited_nodes    =  []
    stack            =  util.Stack()

    # Esse cara aqui armazena em um dicion??rio qual n?? ?? o pai de um determinado n??.
    # Por exemplo: parent[4,5] = (5,5) no layout tiny maze.
    parent           =  {}

    # Armazena quem s??o os vizinhos de um determinado n?? N
    neighbor_nodes   =  []

    is_goal_found    =  False    
    neighbor_nodes   =  problem.getSuccessors(current_pos)

    # Marca o n?? inicial como visitado
    visited_nodes.append(current_pos)

    # Como o n?? inicial n??o tem pai, o n?? pai dele ?? ele mesmo :P
    parent[ problem.getStartState() ] = problem.getStartState()

    # Populando a queue para simplificar o la??o while abaixo
    for i in neighbor_nodes:
        stack.push(i)
        parent[ i[0] ] = current_pos
    
    while stack and not is_goal_found:
        current_pos = stack.pop()[0]
        visited_nodes.append(current_pos)
        
        # Se a posi????o atual for o objetivo
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        # Se n??o for, pegar os vizinhos do n?? atual, adicionar os n??o-visitados
        # na queue e continuar o la??o while.
        neighbor_nodes = problem.getSuccessors(current_pos)
        
        for i in neighbor_nodes:
            is_in_visited_nodes = False
            for j in visited_nodes:
                if i[0] == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                stack.push(i)
                parent[ i[0] ] = current_pos
                
    # Para saber o caminho, o melhor jeito ?? usar nosso dicion??rio que armazena o n?? pai de outro n??.
    # Essa estrat??gia ?? boa no BFS porque pelo funcionamento do algoritmo, navegar pelos n??-pai vai sempre 
    # nos dar a menor dist??ncia entre dois n??s. No DFS isso j?? n??o funcionaria, por exemplo.
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node]

    # Aqui eu come??o no n?? que ?? o final do labirinto e vou navegando de n??-filho para n??-pai at?? chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as dire????es e armazeno elas na vari??vel solution
    while child_node != problem.getStartState():
        for neighbor, direction, cost in problem.getSuccessors(parent_node):
            if neighbor == child_node:
                solution.insert(0, direction)
                child_node  = parent_node
                parent_node = parent[parent_node]
                break

    return solution


def breadthFirstSearch(problem: SearchProblem):

    # Sempre come??a no n?? inicial
    current_pos      =  problem.getStartState()

    visited_nodes    =  []
    queue            =  util.Queue()

    # Esse cara aqui armazena em um dicion??rio qual n?? ?? o pai de um determinado n??.
    # Por exemplo: parent[4,5] = (5,5) no layout tiny maze.
    parent           =  {}

    # Armazena quem s??o os vizinhos de um determinado n?? N
    neighbor_nodes   =  []

    is_goal_found    =  False    
    neighbor_nodes   =  problem.getSuccessors(current_pos)

    # Marca o n?? inicial como visitado
    visited_nodes.append(current_pos)

    # Como o n?? inicial n??o tem pai, o n?? pai dele ?? ele mesmo :P
    parent[ problem.getStartState() ] = problem.getStartState()

    # Populando a queue para simplificar o la??o while abaixo
    for i in neighbor_nodes:
        queue.push(i)
        parent[ i[0] ] = current_pos
    
    while queue and not is_goal_found:
        current_pos = queue.pop()[0]
        visited_nodes.append(current_pos)
        
        # Se a posi????o atual for o objetivo
        if problem.isGoalState(current_pos):
            is_goal_found = True
            break

        # Se n??o for, pegar os vizinhos do n?? atual, adicionar os n??o-visitados
        # na queue e continuar o la??o while.
        neighbor_nodes = problem.getSuccessors(current_pos)
        
        for i in neighbor_nodes:
            is_in_visited_nodes = False
            for j in visited_nodes:
                if i[0] == j:
                    is_in_visited_nodes = True

            if not is_in_visited_nodes:
                queue.push(i)
                parent[ i[0] ] = current_pos
    

    # Para saber o caminho, o melhor jeito ?? usar nosso dicion??rio que armazena o n?? pai de outro n??.
    # Essa estrat??gia ?? boa no BFS porque pelo funcionamento do algoritmo, navegar pelos n??-pai vai sempre 
    # nos dar a menor dist??ncia entre dois n??s. No DFS isso j?? n??o funcionaria, por exemplo.
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node]

    # Aqui eu come??o no n?? que ?? o final do labirinto e vou navegando de n??-filho para n??-pai at?? chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as dire????es e armazeno elas na vari??vel solution
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

    # Tem o mesmo prop??sito que o <parent> do BFS, mas aqui ele armazena um segundo valor: o gasto total para chegar
    # em um n?? N a partir do n?? inicial
    parent         =  {}

    # Items com prioridade maior ficam mais no final! Items com prioridade menos aparecem nas primeiras posi????es
    # (Priority, Number of Insertion, Item)
    p_queue        =  util.PriorityQueue()

    is_goal_found  =  False

    # Marca o n?? inicial como visitado
    visited_nodes.append(current_pos)

    # Como o n?? inicial n??o tem pai, o n?? pai dele ?? ele mesmo, e obviamente, com custo zero
    parent[ problem.getStartState() ] = [ problem.getStartState(), 0 ]

    # Pega os n??s vizinhos
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
                # priority = step_cost do n?? atual + custo total do caminho at?? o n?? <position>
                p_queue.push(position, step_cost + previous_cost)
                parent[ position ] = [ current_pos, step_cost + previous_cost ]

        current_pos = p_queue.pop()

    # Para saber o caminho, o melhor jeito de novo ?? usar nosso dicion??rio que armazena o n?? pai de outro n??.
    # Dessa forma, vamos seguir precisamente o caminho que o UCS acima percorreu pra achar o goalState
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node][0]    
    
    # Aqui eu come??o no n?? que ?? o final do labirinto e vou navegando de n??-filho para n??-pai at?? chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as dire????es e armazeno elas na vari??vel solution
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


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):

    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # O c??digo do A* ?? literalmente a mesma coisa que o UCS, s?? que ele soma o valor da heur??stica junto ao step_cost
    # e ao custo acumulado do caminho na p_queue. Por isso eu s?? copiei e colei meu c??digo do UCS, e quando eu fa??o um push
    # pra p_queue, eu somo junto o valor da heur??stica.

    current_pos    =  problem.getStartState()
    visited_nodes  =  []

    # Tem o mesmo prop??sito que o <parent> do BFS, mas aqui ele armazena um segundo valor: o gasto total para chegar
    # em um n?? N a partir do n?? inicial
    parent         =  {}

    # Items com prioridade maior ficam mais no final! Items com prioridade menos aparecem nas primeiras posi????es
    # (Priority, Number of Insertion, Item)
    p_queue        =  util.PriorityQueue()

    is_goal_found  =  False

    # Marca o n?? inicial como visitado
    visited_nodes.append(current_pos)

    # Como o n?? inicial n??o tem pai, o n?? pai dele ?? ele mesmo, e obviamente, com custo zero
    parent[ problem.getStartState() ] = [ problem.getStartState(), 0 ]

    # Pega os n??s vizinhos
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
                # priority = step_cost do n?? atual + custo total do caminho at?? o n?? <position>
                p_queue.push(position, step_cost + previous_cost + heuristic(current_pos, problem))
                parent[ position ] = [ current_pos, step_cost + previous_cost + heuristic(current_pos, problem) ]

        current_pos = p_queue.pop()

    # Para saber o caminho, o melhor jeito de novo ?? usar nosso dicion??rio que armazena o n?? pai de outro n??.
    # Dessa forma, vamos seguir precisamente o caminho que o UCS acima percorreu pra achar o goalState
    solution     =  []
    child_node   =  current_pos
    parent_node  =  parent[child_node][0]    
    
    # Aqui eu come??o no n?? que ?? o final do labirinto e vou navegando de n??-filho para n??-pai at?? chegar onde era o ponto inicial. 
    # Assim eu consigo pegar as dire????es e armazeno elas na vari??vel solution
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