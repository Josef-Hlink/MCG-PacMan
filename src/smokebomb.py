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
from typing import Literal

import math
from captureAgents import CaptureAgent
from capture import GameState
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
    def registerInitialState(self, gameState: GameState) -> None:
        CaptureAgent.registerInitialState(self, gameState)

        # number of agents in the game
        self.nAgents: int = gameState.getNumAgents()
        # time limit for the MCTS part of the algorithm (per turn)
        self.timeLimit: float = .9
        # maximum depth of the rollout phase
        self.rolloutDepth: int = 200
        # for each iteration in the rollout phase, the agents will move in this order
        self.moveOrder: list[int] = [i for i in range(self.index, self.nAgents)] + [i for i in range(self.index)]
        # list of opponent indices
        self.opponents: list[int] = self.getOpponents(gameState)
        # maze distances between all pairs of positions
        self.distances: dict[tuple, dict[tuple, int]] = self.getDistanceHashTable(gameState)
        # position of the agent's home
        self.homePos: tuple[int, int] = gameState.getAgentPosition(self.index)
        # list of positions in the agent's home
        self.territory = self.getTerritory(gameState)
        # list of positions just outside of opponent's territory
        self.safeStrip = self.getSafeStrip(gameState)
        # mode of the agent (not used in MCTS & UCT)
        global FORAGE, CHASE, RUN
        FORAGE, CHASE, RUN = 'f', 'c', 'r'
        self.mode: Literal['f', 'c', 'r'] = None

    def chooseAction(self, gameState: GameState) -> str:
        
        startTime = time.perf_counter()
        root = Node(gameState)
        i = 0
        maxDepth = 0

        while time.perf_counter() - startTime < self.timeLimit:
            i += 1
            node = root

            # SELECTION
            depth = 0
            while node.c:
                bNode, bVal = None, -math.inf
                for child in node.c:
                    val = self.selectionValue(child)
                    if val > bVal:
                        bNode, bVal = child, val

                depth += 1
                node = bNode
            
            maxDepth = max(maxDepth, depth)

            # EXPANSION
            for action in self.legalActions(node.s):
                child = Node(node.s.generateSuccessor(self.index, action), node, action)
                node.c.append(child)

            # SIMULATION
            node = random.choice(node.c)
            val = self.rollout(node)

            # BACKPROP
            while node:
                node.n += 1
                node.v += val
                node = node.p

        bAct, bVal = random.choice(root.c).a, -math.inf
        if self.index == 0: print('\n' + '-' * 80 + '\n' + ' ' * 35 + f't = {int(300 - gameState.data.timeleft/4)}' + '\n' + '-' * 80)
        else: print()
        if self.mode is not None: print(f'mode: \033[1m{self.mode.capitalize()}\033[0m')
        for child in root.c:
            res = self.finalValue(child)
            if type(res) == float:
                val = res
                print(f'> {child.a:<5} | {val:.4f}')
            else:
                val = res[0] + res[1]
                print(f'> {child.a:<5} | {val:.4f} ({res[0]:.4f} + {res[1]:.4f})')
            if val > bVal:
                bAct, bVal = child.a, val

        self.logAction(bAct)
        endTime = time.perf_counter()
        print(f'{i} iterations (max depth {maxDepth}) took {endTime - startTime:.3f}s with best value {bVal:.4f}')

        return bAct

    def rollout(self, node: Node) -> float:
        """ Will be overwritten by Master Agent. """
        return self._randomRollout(node)

    def selectionValue(self, node: Node) -> float:
        """ Will be overwritten by both UCT Agent and Master Agent. """
        return self._pureValue(node)

    def finalValue(self, node: Node) -> float:
        """ Will be overwritten by Master Agent. """
        return self._pureValue(node)

    # methods to be called on initialization

    def getDistanceHashTable(self, gameState: GameState) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
        """ Returns a hash table of distances between all possible combinations of positions. """
        walls = gameState.data.layout.walls
        positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        return {pos1: {pos2: self.getMazeDistance(pos1, pos2) for pos2 in positions} for pos1 in positions}

    def getSafeStrip(self, gameState: GameState) -> list[tuple[int, int]]:
        """
        Returns a list of positions on the safe strip.
        The safe strip is the strip (column) of positions that are closest to the opponent's side.
        """
        territory = self.getTerritory(gameState)
        edgeCol = max(territory, key=lambda pos: pos[0])[0] if self.red else min(territory, key=lambda pos: pos[0])[0]
        return [pos for pos in territory if pos[0] == edgeCol]

    def getTerritory(self, gameState: GameState) -> tuple[int, int]:
        """ Returns all of the positions that are inside the agent's territory. """
        walls = gameState.data.layout.walls
        positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        redPositions = [pos for pos in positions if pos[0] < walls.width / 2]
        bluePositions = [pos for pos in positions if pos[0] >= walls.width / 2]
        return redPositions if self.red else bluePositions

    # methods to be called multiple times

    def legalActions(self, gameState: GameState) -> list[str]:
        """ Returns a list of legal actions for the agent (without STOP). """
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        return legalActions

    def getTeamMatePos(self, gameState: GameState) -> tuple[int, int]:
        """ Returns the position of the agent's team mate. """
        return gameState.getAgentPosition(self.getTeam(gameState)[1 - self.index % 2])

    def closestDistToSafeStrip(self, gameState: GameState) -> int:
        """ Returns the distance to the closest safe strip position. """
        return min(self.distances[gameState.getAgentPosition(self.index)][pos] for pos in self.safeStrip)

    def evaluateRollout(self, gameState: GameState) -> float:
        """ Checks how much food is left for the agent and the opponent. """
        oppFoodLeft = self.getFoodYouAreDefending(gameState).count(True)
        foodLeft = self.getFood(gameState).count(True)
        return oppFoodLeft - foodLeft + gameState.getScore() * 2
 
    def _pureValue(self, node: Node) -> float:
        """ Average observed value of a node. """
        if node.n == 0: return math.inf
        return node.v / node.n

    def _randomRollout(self, fromNode: Node) -> float:
        """ Randomly plays out the game from the given node until the game is over or the rollout depth is reached. """
        state, depth = fromNode.s, 0
        while not state.isOver() and depth < self.rolloutDepth:
            for i in self.moveOrder:
                actions = state.getLegalActions(i)
                actions.remove(Directions.STOP)
                state = state.generateSuccessor(i, random.choice(actions))
            depth += 1
        return self.evaluateRollout(state)

    def logAction(self, action: str) -> None:
        """ Prints the action taken by the agent in a nice way. """
        currentPos = self.getCurrentObservation().getAgentPosition(self.index)
        dX, dY = Actions.directionToVector(action)
        nextPos = (int(currentPos[0] + dX), int(currentPos[1] + dY))
        print(f'{self.__class__.__name__}({self.index}): {currentPos} -> {action} -> {nextPos}')

class UCTAgent(MCTSAgent):
    """
    Extremely similar to MCTSAgent,
    but uses UCB1 formula for tree traversal in selection phase.
    """
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

    def selectionValue(self, node: Node) -> float:
        """ UCB1 formula for tree traversal. """
        return self._UCB1Value(node)

    def _UCB1Value(self, node: Node) -> float:
        """ UCB1 formula for tree traversal. """
        if node.n == 0: return math.inf
        return node.v / node.n + math.sqrt(2 * math.log(node.p.n) / node.n)

class MasterAgent(UCTAgent):
    """
    Basically the UCT agent but with all sorts of cool heuristics.
    """
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.rolloutDepth = 100
        self.timeLimit = 0.9

    def chooseAction(self, gameState: GameState) -> str:

        # when evaluating children in _finalHeuristicValue,
        # we want to know some things about the current game state (their parent)
        # these variables will be prefixed with "p" to indicate that they refer to the parent's state
        # this "p" might also be confused with "previous", and that is actually fine semantically too
        
        # states
        self.pGameState = gameState; del gameState
        self.pAgentState = self.pGameState.getAgentState(self.index)
        self.pIsPacman = self.pAgentState.isPacman
        # our positions
        self.pPos = self.pGameState.getAgentPosition(self.index)
        self.pTeamMatePos = self.getTeamMatePos(self.pGameState)
        # food
        self.pFoodPositions = sorted(self.getFood(self.pGameState).asList(), key=lambda pos: self.distances[self.pPos][pos])
        self.pNumCarrying = self.pAgentState.numCarrying
        self.pNumReturned = self.pAgentState.numReturned
        # enemies
        pEnemyPositions = [self.pGameState.getAgentPosition(eIndex) for eIndex in self.getOpponents(self.pGameState)]
        self.pEnemyDistances = sorted([self.distances[self.pPos][pos] for pos in pEnemyPositions])
        self.pEnemyPositions = sorted(pEnemyPositions, key=lambda pos: self.distances[self.pPos][pos])
        pEnemiesInTerritory = [pos for pos in self.pEnemyPositions if pos in self.territory]
        self.pEnemiesInTerritory = sorted(pEnemiesInTerritory, key=lambda pos: self.distances[self.pPos][pos])
        self.pEnemiesScared = self.pGameState.getAgentState(self.getOpponents(self.pGameState)[0]).scaredTimer > 0

        self.setMode()

        return super().chooseAction(self.pGameState)

    def setMode(self) -> None:
        """ Sets the mode of the agent. """
        # if there is an enemy in our territory and we are closer to them than our teammate is,
        # we want to chase them down
        if (
            (self.pEnemiesInTerritory and
            self.distances[self.pPos][self.pEnemiesInTerritory[0]] <=
            self.distances[self.pTeamMatePos][self.pEnemiesInTerritory[0]] + 2) or
            (self.pIsPacman and self.pEnemiesScared)
        ):
            self.mode = CHASE
        # if we are carrying a lot of food or the enemy is close (while we are in their territory),
        # we want to run away from them to the safe strip
        elif (
            self.pNumCarrying >= 3 or
            (self.pIsPacman and self.pEnemyDistances[0] <= 4)
        ):
            self.mode = RUN
        # in all other situations, just look for food
        else:
            self.mode = FORAGE

    def _finalHeuristicValue(self, node: Node) -> float:
        cGameState = node.s
        cPos = cGameState.getAgentPosition(self.index)
        cAgentState = cGameState.getAgentState(self.index)
        cNumCarrying = cAgentState.numCarrying

        val = 0
        # if we are in foraging mode
        if self.mode == FORAGE:
            # we want to maximize the number of food pellets we have
            val += (cNumCarrying - self.pNumCarrying) * 2
            # we also want to minimize the distance to the closest food pellet
            val -= self.distances[cPos][self.pFoodPositions[0]] * 1
        # if we are in running mode
        elif self.mode == RUN:
            # we want to maximize the distance to the closest enemy
            val += self.distances[cPos][self.pEnemyPositions[0]] * 2
            # we also want to minimize the distance to the closest safe strip position
            val -= self.closestDistToSafeStrip(cGameState) * .5
        # if we are in chasing mode
        elif self.mode == CHASE:
            # we want to minimize the distance to the closest enemy
            val -= self.distances[cPos][self.pEnemiesInTerritory[0]] * 2
            # we also want to minimize the distance to the closest safe strip position
            # so in case of a tie, we will choose the position that is closer to where they want to go
            val -= self.closestDistToSafeStrip(cGameState) * .5
        else:
            raise Exception(f'Unknown mode: {self.mode}')

        return val

    def selectionValue(self, node: Node) -> float:
        """ UCB1 formula for tree traversal. """
        return self._UCB1Value(node)

    def finalValue(self, node: Node) -> float | tuple[float, float]:
        """ Combination of pure value plus heuristic evaluation of a node. """
        return self._pureValue(node), self._finalHeuristicValue(node)

    def rollout(self, fromNode: Node) -> float:
        """ Heuristic evaluation of a node. """
        # TODO: implement this
        return self._randomRollout(fromNode)
