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

    def __init__(self, index):
        super().__init__(self, index)

    def chooseAction(self, gameState: GameState) -> str:
        """
        Picks among actions randomly.
        """

        return self.monteCarloTreeSearch(gameState)


    def monteCarloTreeSearch(self, gameState: GameState) -> str:
        """
        Performs a Monte Carlo Tree Search on the given state and returns the best action.
        """
        
        actions = gameState.getLegalActions(self.index)
        Q: np.ndarray = np.zeros(len(actions))
        for i, action in enumerate(actions):
            Q[i] += self.monteCarloPlayOut(gameState, action)
        return actions[np.argmax(Q)]

    def monteCarloPlayOut(self, gameState: GameState, action: str, maxDepth: int = 50) -> bool:
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
            return True
        else:
            return True
