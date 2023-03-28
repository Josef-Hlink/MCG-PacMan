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


import random, time, util
from datetime import datetime

import numpy as np
import math
from captureAgents import CaptureAgent, RandomAgent
from capture import GameState
from game import Directions

#################
# Team creation #
#################

def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first = 'UCTAgent',
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


class UCTAgent(CaptureAgent):
    def registerInitialState(self, gameState: GameState):
        # The following line implicitly sets the agent's `self.index`.
        CaptureAgent.registerInitialState(self, gameState)
        self.timeLimit = 0.9     # amount of time the agent has to come up with an action
        self.c = math.sqrt(2)    # exploration constant
        self.rolloutDepth = 100  # maximum depth of a rollout

    def chooseAction(self, gameState: GameState) -> str:
        # THIS VARIABLE WILL CONTAIN ALL THE INFORMATION ABOUT THE GAME TREE
        # AND SHOULD BE HANDLED WITH CARE, SO BE AWARE TO NOT OVERWRITE IT SOMEHOW
        root = Node(gameState)

        # Time limit for the agent to select the best-so-far action.
        startTime = time.perf_counter()
        
        # If for some weird reason we are never allowed to iterate, we should just skip a move.
        # This should not happen, but it prevents this method from raising a NameError.
        bAct = 'Stop'
        
        # Start looking boy...
        while time.perf_counter() - startTime < self.timeLimit:
            
            # SIGNIFIES THE CURRENT NODE WE ARE WORKING WITH IN THE GAME TREE
            # ALWAYS START WITH THE ROOT NODE, AND OVERWRITE WITH BEST LEAF NODE
            node = root

            # --------------- #
            # selection phase #
            # --------------- #
            while node.c:                           # while node is not a leaf node (it has children)
                bNode, bVal = None, -math.inf       # initialize best node and best value to nonsense values
                for child in node.c:                # loop through all children of the current node
                    val = self.UCT(child)           # calculate the UCT value of the child
                    if val > bVal:                  # if the value is better than the best value so far
                        bNode, bVal = child, val    # update the best node and best value
                
                # NOW THE BEST LEAF IS SELECTED, SO WE CAN MOVE TO THAT NODE
                node = bNode

            # Okay, now we have a leaf node, we can finally move on with the cool stuff.

            if node.n == 0:
                # ---------------- #
                # simulation phase #
                # ---------------- #
                rolloutVal = self.rollout(node.s)
            else:
                # --------------- #
                # expansion phase #
                # --------------- #
                legalActions = node.s.getLegalActions(self.index)
                legalActions.remove('Stop')
                for action in legalActions:
                    newState = node.s.generateSuccessor(self.index, action)
                    child = Node(newState, node, action)
                    node.c.append(child)
                # while we're at it, we do a simulation from one of the children
                node = random.choice(node.c)
                rolloutVal = self.rollout(node.s)

            # -------------- #
            # backprop phase #
            # -------------- #
            while node:
                node.n += 1
                node.v += rolloutVal
                node = node.p

        # choose the best action
        bAct, bVal = None, -math.inf
        for child in root.c:
            val = child.v / child.n
            if val > bVal:
                bAct, bVal = child.a, val
        self.log(bAct)
        return bAct
    

    def UCT(self, node: Node) -> float:
        """ Just the UCT formula, nothing special. """
        if node.n == 0: return math.inf
        return node.v / node.n + self.c * math.sqrt(math.log(node.p.n) / node.n)


    def rollout(self, gameState: GameState) -> float:
        """
        Does a rollout from the given game state and returns the value of the rollout.
        Value is how many food pellets the opponent has left MINUS how many food pellets we have left.
        Each team starts with 20 food pellets.
        
        Examples:
            - If they have eaten 10 pellets and we have eaten 5 pellets, the value is -5.
            - If they have eaten 0 and we have eaten 20, the value is +20.
        """
        state, depth = gameState, 0
        while not state.isOver() and depth < self.rolloutDepth:
            state = state.generateSuccessor(self.index, random.choice(state.getLegalActions(self.index)))
            depth += 1
        return self.evaluate(state)

    def evaluate(self, gameState: GameState) -> float:

        oppFoodLeft = self.getFoodYouAreDefending(gameState).count(True)
        foodLeft = self.getFood(gameState).count(True)
        return oppFoodLeft - foodLeft

    def log(self, action: str) -> None:
        """ Print the action along with a bunch of useful logging information to the console. """
        pIdx, pPos = self.index, self.getCurrentObservation().getAgentPosition(self.index)
        nextPos = {
            'North': (pPos[0], pPos[1] + 1),
            'South': (pPos[0], pPos[1] - 1),
            'East': (pPos[0] + 1, pPos[1]),
            'West': (pPos[0] - 1, pPos[1])
        }[action]
        currentScore = self.evaluate(self.getCurrentObservation())
        print(f'Agent {pIdx} {pPos} -> {action} -> {nextPos}  |  score: {currentScore}')
