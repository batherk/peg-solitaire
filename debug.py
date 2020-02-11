from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
from Main import *
import matplotlib.pyplot as plt

def print_expected_reward_for_win_states(critic):
    for state in critic.expected:
        if state.count('1') == 1:
            print(f'State: {state} Expected future reward: {critic.expected[state]}')

def print_expected_reward_for_states(critic):
    for state in critic.expected:
        print(f'State: {state} Expected future reward: {critic.expected[state]}')


c = create_critic(lam,alpha,gamma,False,15,[3])

tensor = torch.randn(15)

print(tensor)

print(c.net(tensor))