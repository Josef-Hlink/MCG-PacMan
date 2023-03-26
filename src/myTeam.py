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
    first = 'GangsterAgent',
    second = 'GangsterAgent'
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

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)


class GangsterAgent(CaptureAgent):
    
    def registerInitialState(self, gameState: GameState) -> None:
        CaptureAgent.registerInitialState(self, gameState)

        self.startNode: TreeNode = TreeNode(gameState, 'Stop', self, None)
        


    def chooseAction(self, gameState: GameState) -> str:
        """
        Picks among actions randomly.
        """

        actions = gameState.getLegalActions(self.index)
        if len(actions) == 1:
            return actions[0]


        if gameState.getAgentPosition(self.index)[0] == 30:
            if gameState.getAgentPosition(self.index)[1] == 1:
                return 'West'
            return 'South'

        # action = self.basicMonteCarlo(gameState, 25)
        action = self.MCTS(gameState, 25)

        # food left and food you are defending
        foodLeft = self.getFood(gameState).count(True)
        oppFoodLeft = self.getFoodYouAreDefending(gameState).count(True)
        pos = gameState.getAgentPosition(self.index)
        print(f'{str(datetime.now())[:-4]}: Agent {self.index} is on position {pos} and will go {action} | current score: {foodLeft} - {oppFoodLeft}')


        return action

    def MCTS(self, gameState: GameState, n: int) -> str:
        """
        Performs a Monte Carlo tree search on the given state and returns the best action.
        """
        
        self.startNode.expand()
        for _ in range(n):
            node = self.startNode.select()
            node.expand()
            node = node.select()
            node.backpropagate(node.rollout(100))
        

    def basicMonteCarlo(self, gameState: GameState, n: int) -> str:
        """
        Performs a basic Monte Carlo random search on the given state and returns the best action.
        """
        
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        Q: np.ndarray = np.zeros(len(actions))
        for i, action in enumerate(actions):
            for _ in range(n):
                Q[i] += self.randomPlayout(gameState, action)
        print(Q)
        return actions[argmax(Q)]

    def randomPlayout(self, gameState: GameState, action: str, maxDepth: int = 400) -> bool:
        """
        Plays out a random game from the given state and returns True if the agent's team won.
        """

        simulation = gameState.deepCopy()
        d = 0
        while d < maxDepth:
            foodLeft = self.getFood(simulation).count(True)
            if not foodLeft:
                return True
            actions = simulation.getLegalActions(self.index)
            action = random.choice(actions)
            simulation = simulation.generateSuccessor(self.index, action)
            if simulation.isOver():
                break
            d += 1
        
        oppFoodLeft = self.getFoodYouAreDefending(simulation).count(True)
        if oppFoodLeft < foodLeft:
            return False
        elif oppFoodLeft == foodLeft:
            return False
        else:
            return True

class TreeNode:
    def __init__(self, gameState: GameState, player: CaptureAgent, parent: 'TreeNode' = None) -> None:
        self.gameState = gameState
        self.player = player
        self.parent = parent
        self.children: dict[str, TreeNode] = {}
        self.visits = 0
        self.Q = 0

    def expand(self) -> None:
        actions = self.gameState.getLegalActions(self.player.index)
        actions.remove('Stop')
        for action in actions:
            self.children[action] = TreeNode(self.gameState.generateSuccessor(self.player.index, action), action, self)

    def isFullyExpanded(self) -> bool:
        return len(self.children) == len(self.gameState.getLegalActions(self.player.index))

    @property
    def bestChild(self) -> 'TreeNode':
        return max(self.children, key=lambda node: node.Q / node.visits + np.sqrt(2 * np.log(self.visits) / node.visits))

    def select(self) -> 'TreeNode':
        if self.isFullyExpanded():
            return self.bestChild.select()
        else:
            return self

    def rollout(self, maxDepth: int) -> bool:
        simulation = self.gameState.deepCopy()
        d = 0
        while d < maxDepth:
            foodLeft = self.player.getFood(simulation).count(True)
            if not foodLeft:
                return True
            actions = simulation.getLegalActions(self.player.index)
            action = random.choice(actions)
            simulation = simulation.generateSuccessor(self.player.index, action)
            if simulation.isOver():
                break
            d += 1
        
        oppFoodLeft = self.player.getFoodYouAreDefending(simulation).count(True)
        if oppFoodLeft < foodLeft:
            return False
        elif oppFoodLeft == foodLeft:
            return False
        else:
            return True

    def backpropagate(self, reward: bool) -> None:
        self.visits += 1
        self.Q += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def __repr__(self) -> str:
        return f'TreeNode({self.gameState}, {self.action}, {self.parent})'

    def __hash__(self) -> int:
        return hash(self.gameState)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.gameState == other.gameState




def argmax(x: np.ndarray) -> int:
    """ Argmax with random tie-breaking. """
    return random.choice(np.nonzero(x == np.amax(x))[0])

