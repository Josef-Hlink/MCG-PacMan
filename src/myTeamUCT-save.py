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

class UCTNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.total_reward = 0
        self.visit_count = 0
        
    
    def expand(self, actions, index):
        for action in actions:
            child_state = self.state.generateSuccessor(index, action)
            child_node = UCTNode(child_state, self, action)
            self.children.append(child_node)
    
    def get_best_child(self, exploration_parameter):
        best_child = None
        best_score = float('-inf')
        for child in self.children:
            if child.visit_count == 0:
                return child
            score = child.total_reward / child.visit_count + \
                    exploration_parameter * math.sqrt(math.log(self.visit_count) / child.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def update(self, reward):
        self.total_reward += reward
        self.visit_count += 1
        
# def update_node_count(node_count):
#     self.node_count = node_count

class UCTAgent(CaptureAgent):    
    def registerInitialState(self, gameState: GameState) -> None:
        # global root_node
        global root_node1, root_node2
        CaptureAgent.registerInitialState(self, gameState)
        self.exploration_parameter = 1 / math.sqrt(2)
        self.num_iterations = 10
        # root_node = UCTNode(gameState)
        if self.index in (0,1):
            root_node1 = UCTNode(gameState)
        else:
            root_node2 = UCTNode(gameState)
        
        self.save_node = None
    
    # def set_root_node(self, gameState: GameState):
    #     self.root_node = UCTNode(gameState)
        
    # def is_terminal(self, state: GameState) -> bool:
    #     redFoodLeft = state.getRedFood().count(True)
    #     blueFoodLeft = state.getBlueFood().count(True)
    #     return (not redFoodLeft or not blueFoodLeft) or not state.data.timeleft
    
    def chooseAction(self, gameState: GameState):
        # root_node = UCTNode(gameState)
        # traverse the tree to find the node with the gameState
        
        for i in range(self.num_iterations):              
            if not self.save_node:
                if self.index in (0,1):
                    node = root_node1
                else:
                    node = root_node2
            else:
                node = self.save_node
            
            state = gameState.deepCopy()
            
            # Selection phase
            while not state.isOver() and node.children:
                # print("Simulate")
                node = node.get_best_child(self.exploration_parameter)
                action = node.action
                state = state.generateSuccessor(self.index, action)
                # prev_node = node
            
            # Expansion phase
            if not node.children and not state.isOver():
                # print("Expand")
                actions = state.getLegalActions(self.index)
                actions.remove("Stop")
                node.expand(actions, self.index)
                node = random.choice(node.children)
                action = node.action
                state = state.generateSuccessor(self.index, action)
            
            # Simulation phase
            while not state.isOver():
                actions = state.getLegalActions(self.index)
                actions.remove("Stop")
                action = random.choice(actions)
                state = state.generateSuccessor(self.index, action)
            
            # Backpropagation phase
            # reward catching ghost
            reward = 20 - state.data.score
            # print("reward: ", reward)
            while node:
                node.update(reward)
                node = node.parent
    
        # Choose best action
        if not self.save_node:
            if self.index in (0,1):
                # best child
                self.save_node = root_node1.get_best_child(self.exploration_parameter)
                
                # max node
                # self.save_node = max(root_node1.children, key=lambda node: node.visit_count)
            else:
                # best child
                self.save_node = root_node2.get_best_child(self.exploration_parameter)
                
                #max node
                # self.save_node = max(root_node2.children, key=lambda node: node.visit_count)
        else:
            # best child
            self.save_node = self.save_node.get_best_child(self.exploration_parameter)
            
            # max node
            # self.save_node = max(self.save_node.children, key=lambda node: node.visit_count)
        
        best_action = self.save_node.action
        
        # best_action = max(root_node.children, key=lambda node: node.visit_count).action
        # food left and food you are defending
        foodLeft = self.getFood(gameState).count(True)
        oppFoodLeft = self.getFoodYouAreDefending(gameState).count(True)
        pos = gameState.getAgentPosition(self.index)
        print(f'{str(datetime.now())[:-4]}: Agent {self.index} is on position {pos} and will go {best_action} | current score: {foodLeft} - {oppFoodLeft}')
        return best_action
