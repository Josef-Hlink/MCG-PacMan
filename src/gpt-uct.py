import math
import random

class UCTNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.total_reward = 0
        self.visit_count = 0
    
    def expand(self, actions):
        for action in actions:
            child_state = self.state.get_next_state(action)
            child_node = UCTNode(child_state, self, action)
            self.children.append(child_node)
    
    def get_best_child(self, exploration_parameter):
        best_child = None
        best_score = float('-inf')
        for child in self.children:
            score = child.total_reward / child.visit_count + \
                    exploration_parameter * math.sqrt(math.log(self.visit_count) / child.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def update(self, reward):
        self.total_reward += reward
        self.visit_count += 1
    
class UCT:
    def __init__(self, exploration_parameter=1 / math.sqrt(2)):
        self.exploration_parameter = exploration_parameter
    
    def search(self, initial_state, num_iterations):
        root_node = UCTNode(initial_state)
        for i in range(num_iterations):
            node = root_node
            state = initial_state.copy()
            
            # Selection phase
            while not state.is_terminal() and node.children:
                node = node.get_best_child(self.exploration_parameter)
                action = node.action
                state.apply_action(action)
            
            # Expansion phase
            if not node.children and not state.is_terminal():
                actions = state.get_legal_actions()
                node.expand(actions)
                node = random.choice(node.children)
                action = node.action
                state.apply_action(action)
            
            # Simulation phase
            while not state.is_terminal():
                actions = state.get_legal_actions()
                action = random.choice(actions)
                state.apply_action(action)
            
            # Backpropagation phase
            reward = state.get_reward()
            while node:
                node.update(reward)
                node = node.parent
        
        # Choose best action
        best_action = max(root_node.children, key=lambda node: node.visit_count).action
        return best_action
