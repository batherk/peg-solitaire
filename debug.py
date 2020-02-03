from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
import matplotlib.pyplot as plt

lam = 1
alpha = 0.1
gamma = 0.9
epsilon = 0.2


actor = Actor(lam,alpha,gamma,epsilon)
critic = TableCritic(lam,alpha,gamma)

game = PegSolitaire("Triangle",3)
game.show_board()
print(game.get_possible_actions())