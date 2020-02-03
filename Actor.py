from PegSolitaire import *
import numpy as np
import random
   
class Actor:
    """ Chooses best action based on states.  Maps (state,action) to values using dictionary. Chooses action with highest value based on epsilon"""

    def __init__(self, lam, alpha, gamma, epsilon):
        self.policy = {}
        self.trace = {}
        self.alpha = alpha
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_max_action(self, state):
        return max(self.policy[state], key=self.policy[state].get) 

    def get_random_action(self, state):
        return random.choice(list(self.policy[state].keys()))

    def get_action(self, game):
        state = game.get_state()
        if not state in self.policy:
            actions = game.get_possible_actions()
            self.policy[state] = {}
            for action in actions:
                self.policy[state][action] = 0
        if random.random() < self.epsilon:
            return self.get_random_action(state)
        else: 
            return self.get_max_action(state)

    def update_tables(self, delta, sequence):
        self.trace[sequence[-1]] = 1
        for state,action in sequence:
            self.policy[state][action] += self.alpha * delta * self.trace[(state,action)]
            self.trace[(state,action)] *= self.gamma * self.lam
  
        
            
        


         
            

        