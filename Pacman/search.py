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
    
    """
    Código por Rafael Amauri Diniz Augusto - 651047
    """
    solution = _depthFirstSearch(problem, problem.getStartState(), (-1, -1), [], False)[1].split(" ")

    return solution
    

# Uma função separada é necessária por causa das várias chamadas de recursão
def _depthFirstSearch(problem: SearchProblem, currentPos: tuple, parentNode: tuple, visited_nodes:list , isGoalFound: bool):
    
    # Se o nó atual for o objetivo, não há porque continuar a busca
    if(problem.isGoalState(currentPos)):
        return True, ""

    # Marca o nó atual como visitado para evitar loops infinitos
    visited_nodes.append(currentPos)
    neighborNodes = problem.getSuccessors(currentPos)
    
    isGoalFound = False
    solution = ""
    direction = ""

    # Abre uma recursão para cada nó que não é o nó pai, que não foi visitado e se o objetivo ainda não foi encontrado.
    for i in neighborNodes:
        if (i[0] != parentNode) and (i[0] not in visited_nodes and not isGoalFound):
            # Literalmente a direção. Cada i aparece como "((3, 1), 'West', 1)" por exemplo.
            direction = i[1]

            # Aqui é onde a recursão é aberta para cada nó que é válido nos critérios acima
            isGoalFound, solution = _depthFirstSearch(problem, i[0], currentPos, visited_nodes, isGoalFound)
            
            # Isso aqui é feito porque quando o objetivo é achado na recursão, ele retorna um "", e isso ferra no final quando 
            # for feito um split na string. Infelizmente NÃO tem como tirar!!
            if(solution == ""):
                solution = f"{direction}"
            else:
                solution = f"{direction} {solution}"

    return isGoalFound, f"{solution}"


# TODO: BFS
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    currentPos = problem.getStartState()

    util.raiseNotDefined()


def bfs_pirata(problem: SearchProblem):
    
    neighborNodes = problem.getSuccessors(currentPos)
    priorityQueue = []
    isGoalFound = False

    for i in neighborNodes:
        priorityQueue.append(i)

    while not priorityQueue and not isGoalFound:
        for i in priorityQueue:
            currentPos = i[0]
            
            if problem.isGoalState(currentPos):
                isGoalFound = True
            
    pass


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
