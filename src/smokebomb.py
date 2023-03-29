# myTeam.py
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


import random, time
from functools import wraps

import numpy as np
import math
from captureAgents import CaptureAgent
from capture import GameState, halfGrid
from game import Directions, Actions

#################
# Team creation #
#################

def createTeam(
    firstIndex, secondIndex, isRed,
    first = 'MasterAgent',
    second = 'MasterAgent'
) -> list[CaptureAgent]:
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

########
# Node #
########

class Node:
    def __init__(self, state: GameState, parent: 'Node' = None, action: str = None):
        self.s: GameState = state  # GameState object (very fucking bloated)
        self.p: 'Node' = parent    # parent node
        self.a: str = action       # action to get to this node from previous node (parent)
        self.c: list['Node'] = []  # list of Node objects (the actions to get to these nodes is stored there)
        self.n: int = 0            # number of simulations after this node
        self.v: float = 0          # "quality" of this node


#########
# Timer #
#########

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        startTime = time.perf_counter()
        result = func(*args, **kwargs)
        endTime = time.perf_counter()
        print(f'Runtime: {func.__name__} = {endTime - startTime:.3f}s')
        return result
    return wrapper


##########
# Agents #
##########

class MCTSAgent(CaptureAgent):
    """
    Most basic form of an Monte Carlo Tree Search agent.
    Uses average value (total value / no. visits) for tree traversal in selection phase.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.timeLimit: float = .9
        self.rolloutDepth: int = 100
        self.valueOfNode: function = self.valueOfNodePure
        self.valueOfFinalNode: function = self.valueOfNodePure
        self.distances: dict = self.getDistanceHashTable(gameState)

    def chooseAction(self, gameState: GameState) -> str:
        
        startTime = time.perf_counter()
        root = Node(gameState)
        i = 0

        while time.perf_counter() - startTime < self.timeLimit:
            i += 1
            node = root

            # SELECTION
            while node.c:
                bNode, bVal = None, -math.inf
                for child in node.c:
                    val = self.valueOfNode(child)
                    if val > bVal:
                        bNode, bVal = child, val
    
                node = bNode
            
            # EXPANSION
            for action in self.legalActions(node.s):
                child = Node(node.s.generateSuccessor(self.index, action), node, action)
                node.c.append(child)

            # SIMULATION
            node = random.choice(node.c)
            val = self.rollout(node.s)

            # BACKPROP
            while node:
                node.n += 1
                node.v += val
                node = node.p

        bAct, bVal = random.choice(root.c).a, -math.inf
        for child in root.c:
            val = self.valueOfFinalNode(child)
            if val > bVal:
                bAct, bVal = child.a, val

        if self.index == 0: print(f'\nt = {int(300 - gameState.data.timeleft/4)}')
        self.logAction(bAct)
        endTime = time.perf_counter()
        print(f'i = {i}, runtime = {endTime - startTime:.3f}s, value = {bVal:.3f}')

        return bAct

    def rollout(self, gameState: GameState) -> float:
        state, depth = gameState, 0
        while not state.isOver() and depth < self.rolloutDepth:
            state = state.generateSuccessor(self.index, random.choice(self.legalActions(state)))
            depth += 1
        return self.evaluate(state)
    
    def evaluate(self, gameState: GameState) -> float:
        oppFoodLeft = self.getFoodYouAreDefending(gameState).count(True)
        foodLeft = self.getFood(gameState).count(True)
        return oppFoodLeft - foodLeft
 
    def legalActions(self, gameState: GameState) -> list[str]:
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        return legalActions

    def valueOfNodePure(self, node: Node) -> float:
        if node.n == 0: return math.inf
        return node.v / node.n

    def logAction(self, action: str) -> None:
        """ Prints the action taken by the agent. """
        currentPos = self.getCurrentObservation().getAgentPosition(self.index)
        dX, dY = Actions.directionToVector(action)
        nextPos = (int(currentPos[0] + dX), int(currentPos[1] + dY))
        print(f'{self.__class__.__name__}({self.index}): {currentPos} -> {action} -> {nextPos}')

    def getDistanceHashTable(self, gameState: GameState) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
        """ Returns a hash table of distances between all possible combinations of state positions. """
        walls = gameState.data.layout.walls
        positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        return {pos1: {pos2: self.getMazeDistance(pos1, pos2) for pos2 in positions} for pos1 in positions}

    def getSafeStrip(self, gameState: GameState) -> list[tuple[int, int]]:
        """ Returns a list of positions on the safe strip. """
        homePositions = self.getHomePositions(gameState)
        minmax = max if self.red else min
        return [pos for pos in homePositions if pos[0] == minmax(pos[0] for pos in homePositions)]
    
    def getHomeStrip(self, gameState: GameState) -> list[tuple[int, int]]:
        homePositions = self.getHomePositions(gameState)
        minmax = min if self.red else max
        return [pos for pos in homePositions if pos[0] == minmax(pos[0] for pos in homePositions)]

    def getHomePositions(self, gameState: GameState) -> tuple[int, int]:
        walls = gameState.data.layout.walls
        positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        redPositions = [pos for pos in positions if pos[0] < walls.width / 2]
        bluePositions = [pos for pos in positions if pos[0] >= walls.width / 2]
        if self.red:
            return redPositions
        else:
            return bluePositions


class UCTAgent(MCTSAgent):
    """
    Extremely similar to MCTSAgent,
    but uses UCB1 formula for tree traversal in selection phase.
    """
    def registerInitialState(self, gameState):
        MCTSAgent.registerInitialState(self, gameState)
        self.valueOfNode: function = self.valueOfNodeUCT
        self.valueOfFinalNode: function = self.valueOfNodePure
        self.c: float = math.sqrt(2)

    def valueOfNodeUCT(self, node: Node) -> float:
        if node.n == 0: return math.inf
        return node.v / node.n + self.c * math.sqrt(math.log(node.p.n) / node.n)

class MasterAgent(UCTAgent):
    """
    Basically the UCT agent but with all sorts of cool heuristics.
    """
    def registerInitialState(self, gameState):
        UCTAgent.registerInitialState(self, gameState)
        self.valueOfFinalNode: function = self.valueOfNodeMaster
        self.timeLimit = .9
        # ---- Info on environment ---- #
        self.homePos: tuple[int, int] = gameState.getAgentPosition(self.index)
        self.territory = self.getHomePositions(gameState)
        self.safeStrip = self.getSafeStrip(gameState)
        self.homeStrip = self.getHomeStrip(gameState)
        self.teamMatePos: tuple[int, int] = self.getTeamMatePos(gameState)

        # ---- Heuristic flags ---- #
        self.isHome: bool = True      # True if agent is at home, False if agent is at enemy's side
        self.isChasing: bool = False  # True if agent is actively chasing an enemy, False if agent is defending
        self.isRunning: bool = False  # True if agent is running away from an enemy towards home
        self.isScared: bool = False   # True if agent is scared, False if agent is not scared


    def chooseAction(self, gameState: GameState) -> str:
        """ Returns the action to take based on the current state. """
        # ---- Update positions ---- #
        self.ourPos = gameState.getAgentPosition(self.index)
        self.teamMatePos = self.getTeamMatePos(gameState)
        self.foodPositions = self.getFood(gameState).asList()
        self.enemyPositions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState)]

        # ---- Update flags ---- #
        self.isRunning, self.isChasing = False, False
        self.isInHomeStrip = self.ourPos in self.homeStrip
        self.isInSafeStrip = self.ourPos in self.safeStrip
        
        self.nCarrying: int = gameState.getAgentState(self.index).numCarrying

        closestEnemyPos = min(self.enemyPositions, key=lambda pos: self.distances[self.ourPos][pos])
        closestSafePos = min(self.safeStrip, key=lambda pos: self.distances[self.ourPos][pos])

        if self.nCarrying > 2:
            self.isRunning = True
            self.chasing = False
        elif self.nCarrying > 0 and self.distances[self.ourPos][closestEnemyPos] < min(3, self.distances[self.ourPos][closestSafePos]):
        # elif self.distances[self.ourPos][closestEnemyPos] < min(5, self.distances[self.ourPos][closestSafePos]):
            self.isRunning = True
            self.chasing = False
        else:
            for enemy in self.enemyPositions:
                if enemy in self.territory:
                    # if we are closer to the enemy than our teammate is, we should chase
                    if self.distances[self.ourPos][enemy] <= self.distances[self.teamMatePos][enemy] and min(self.getTeam(gameState)) == self.index:
                        self.isChasing = True
                        self.isRunning = False

        # ---- Choose action ---- #
        action = UCTAgent.chooseAction(self, gameState)
        return action

    def valueOfNodeMaster(self, node: Node) -> float:
        closestDistToSafeStrip = self.closestDistToSafeStrip(node.s)
        closestDistToFood = self.closestDistToFood(node.s)
        
        finalValue = self.valueOfNodeUCT(node)

        if self.isInHomeStrip:
            # just always minimize distance to middle of the field
            finalValue += 100 / closestDistToSafeStrip
            return finalValue

        if self.isChasing:
            # minimize the distance to attacking pacman
            finalValue += 10 / min(self.distToEnemies(node.p.s))
            return finalValue
        
        if self.isRunning:
            # maximize the distance to defending ghost
            finalValue += 10 * min(self.distToEnemies(node.p.s))
            # while also minimizing the distance to the safe zone so it deposits food
            finalValue += 5 / (closestDistToSafeStrip + 0.001)
            return finalValue
        
        # incentivize eating food
        if node.s.getAgentPosition(self.index) in self.foodPositions:
            finalValue += 2
        # incentivize to move closer to food
        else:
            finalValue += 1 / closestDistToFood

        return finalValue

    def getTeamMatePos(self, gameState: GameState) -> tuple[int, int]:
        """ Returns the position of the agent's team mate. """
        teamMateIndex = self.getTeam(gameState)[1 - self.index % 2]
        return gameState.getAgentPosition(teamMateIndex)

    def closestDistToSafeStrip(self, gameState: GameState) -> int:
        """ Returns the distance to the closest safe strip position. """
        return min(self.distances[gameState.getAgentPosition(self.index)][pos] for pos in self.safeStrip)

    def closestDistToFood(self, gameState: GameState) -> int:
        """ Returns the distance to the closest food pellet. """
        return min(self.distances[gameState.getAgentPosition(self.index)][pos] for pos in self.foodPositions)

    def distToEnemies(self, gameState: GameState) -> list[tuple[int, int]]:
        """ Returns the distance to the enemies. """
        return [self.distances[gameState.getAgentPosition(self.index)][gameState.getAgentPosition(i)] for i in self.getOpponents(gameState)]
