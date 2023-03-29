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

import numpy as np
import math
from captureAgents import CaptureAgent
from capture import GameState
from game import Directions, Actions

#################
# Team creation #
#################

def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first = 'MCTSAgent',
    second = 'UCTAgent'
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

##########
# Agents #
##########

# Minimalistic node class for the game tree.
class Node:
    def __init__(self, state: GameState, parent: 'Node' = None, action: str = None):
        self.s: GameState = state  # GameState object (very fucking bloated)
        self.p: 'Node' = parent    # parent node
        self.a: str = action       # action to get to this node from previous node (parent)
        self.c: list['Node'] = []  # list of Node objects (the actions to get to these nodes is stored there)
        self.n: int = 0            # number of simulations after this node
        self.v: float = 0          # "quality" of this node


class MCTSAgent(CaptureAgent):
    """
    Most basic form of an Monte Carlo Tree Search agent.
    Uses average value (total value / no. visits) for tree traversal in selection phase.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.timeLimit: float = .9
        self.rolloutDepth: int = 50
        self.valueOfNode: function = self.valueOfNodePure
        self.valueOfFinalNode: function = self.valueOfNodePure

    def chooseAction(self, gameState: GameState) -> str:
        root = Node(gameState)
        startTime = time.perf_counter()
        i = 0

        while time.perf_counter() - startTime < self.timeLimit:
            i += 1
            node = root

            # SELECTION
            while node.c:
                bNode, bVal = None, math.inf
                for child in node.c:
                    val = self.valueOfNode(child)
                    if val < bVal:
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

        bAct, bVal = Directions.STOP, math.inf
        for child in root.c:
            val = self.valueOfFinalNode(child)
            if val < bVal:
                bAct, bVal = child.a, val

        if self.index == 0: print(f'\nt = {int(300 - gameState.data.timeleft/4)}')
        self.logAction(bAct)

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


class UCTAgent(MCTSAgent):
    """
    Extremely similar to MCTSAgent,
    but uses UCB1 formula for tree traversal in selection phase.
    """
    def registerInitialState(self, gameState):
        MCTSAgent.registerInitialState(self, gameState)
        self.valueOfNode: function = self.valueOfNodeUCT
        self.valueOfFinalNode: function = self.valueOfNodePure

    def valueOfNodeUCT(self, node: Node) -> float:
        if node.n == 0: return math.inf
        return node.v / node.n + math.sqrt(2 * math.log(node.p.n) / node.n)
