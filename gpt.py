import random
import numpy as np
from copy import deepcopy

class Node:
    def __init__(self, parent, state, action):
        self.parent = parent
        self.state = state
        self.action = action
        self.visits = 0
        self.reward = 0
        self.children = []

    def is_fully_expanded(self):
        return len(self.children) == len(self.untried_actions())

    def untried_actions(self):
        actions = self.state.get_legal_actions(self.action)
        for child in self.children:
            if child.action in actions:
                actions.remove(child.action)
        return actions

    def select_child(self, c):
        children = sorted(self.children, key=lambda c: c.reward / c.visits + c.parent_reward(c.visits, c.parent.visits, c.visits))
        return children[-1]

    def expand(self):
        action = random.choice(self.untried_actions())
        state = deepcopy(self.state)
        state.apply_action(self.action, action)
        child = Node(self, state, action)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.reward += reward

    def parent_reward(self, visits, parent_visits, child_visits):
        return 2 * np.sqrt(np.log(parent_visits) / child_visits)

class MCTS:
    def __init__(self, state, max_iter, C):
        self.root = Node(None, state, None)
        self.max_iter = max_iter
        self.C = C

    def run(self):
        for i in range(self.max_iter):
            node = self.root
            while not node.state.is_terminal():
                if not node.is_fully_expanded():
                    child = node.expand()
                    reward = self.rollout(child.state)
                    child.update(reward)
                    break
                else:
                    node = node.select_child(self.C)
            else:
                reward = self.rollout(node.state)
                node.update(reward)

    def best_action(self):
        return max(self.root.children, key=lambda c: c.visits).action

    def rollout(self, state):
        while not state.is_terminal():
            actions = state.get_legal_actions()
            action = random.choice(actions)
            state.apply_action(action)
        return state.get_reward()

class GameState:
    def __init__(self, grid, max_steps, p1_start, p2_start, p1_goal, p2_goal):
        self.grid = grid
        self.max_steps = max_steps
        self.p1_pos = p1_start
        self.p2_pos = p2_start
        self.p1_goal = p1_goal
        self.p2_goal = p2_goal
        self.steps = 0

    def get_legal_actions(self, agent):
        actions = []
        x, y = self.p1_pos if agent == 1 else self.p2_pos
        if self.grid[x][y] == ' ':
            if y > 0 and self.grid[x][y-1] != '%':
                actions.append('West')
            if y < len(self.grid[0])-1 and self.grid[x][y+1] != '%':
                actions.append('East')
            if x > 0 and self.grid[x-1][y] != '%':
                actions.append('North')
            if x < len(self.grid)-1 and self.grid[x+1][y] != '%':
                actions.append('South')
        return actions

   
