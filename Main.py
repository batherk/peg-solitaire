from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
import matplotlib.pyplot as plt


# Scenario: 
SCENARIO = 2

SCENARIO_DESCRIPTIONS = {}
SCENARIO_DESCRIPTIONS[0] = "Custom settings"
SCENARIO_DESCRIPTIONS[1] = "Triangle, 5, Table, (0,0) is empty, 500 epochs, alpha_actor: 0.001, alpha_critic: 0.001."
SCENARIO_DESCRIPTIONS[2] = "Diamond, 4, Table, (1,2) is empty, 1000 epochs, alpha_actor: 0.001, alpha_critic: 0.001." 
SCENARIO_DESCRIPTIONS[3] = "Triangle, 5, Neural Net, (0,0) is empty, 500 epochs, alpha_actor: 0.001, alpha_critic: 0.0001."
SCENARIO_DESCRIPTIONS[4] = "Diamond, 4, Neural Net, (1,2) is empty, 1000 epochs, alpha_actor: 0.001, alpha_critic: 0.0001."

# Parameter settings
lam = 0.5
gamma = 0.9
epsilon = 0.15

alpha_critic = 0.001
alpha_actor = 0.001

# Game settings
board_type = "Triangle"
board_size = 5
empty_nodes_pos = [] # If this is empty, the game initializes with a random peg every time

# Epoch settings
times_with_learning = 1500

# Checking policy
times_after_learning_testing = 500
times_after_learning_with_show_every_move = 1
times_after_learning_with_show_last_move = 0

# Statistics settings
show_statistics = True
points_amount = 100
average_amount = (times_with_learning + times_after_learning_testing + times_after_learning_with_show_every_move + times_after_learning_with_show_last_move)//points_amount

# Plot settings
delay = 0.2

# Debug settings
debug = False

# Critic settings
USE_TABLE = False

# Neural net critic
UPDATE_WHOLE_SEQUENCE = True # If true the net is updated by every state in the sequence, based on eligibility traces. Only one time per state per episode. 
LAYERS = [15,10,10,10,1]  # This works: LAYERS = [X,10,10,10,X], ADAPT_END_LAYERS = True
ADAPT_END_LAYERS = True # If true the first layer of nodes gets the dimension of the state size of the game and last layer gets one node, else custom


# Setting constants for the different scenarios
if SCENARIO == 1:
    USE_TABLE = True
    board_type = "Triangle"
    board_size = 5
    times_with_learning = 500
    alpha_critic = 0.001
    alpha_actor = 0.001
    empty_nodes_pos = [(0,0)]
elif SCENARIO == 2:
    USE_TABLE = True
    board_type = "Diamond"
    board_size = 4
    times_with_learning = 1000
    alpha_critic = 0.001
    alpha_actor = 0.001
    empty_nodes_pos = [(1,2)]
elif SCENARIO == 3:
    USE_TABLE = False
    board_type = "Triangle"
    board_size = 5
    times_with_learning = 500
    alpha_critic = 0.0001
    alpha_actor = 0.001
    empty_nodes_pos = [(0,0)]
elif SCENARIO == 4:
    USE_TABLE = False
    board_type = "Diamond"
    board_size = 4
    times_with_learning = 1000
    alpha_critic = 0.00005
    alpha_actor = 0.001
    empty_nodes_pos = [(1,2)]



def run_one_game(game,actor,critic,evolve=True,show_board_every_action=False, show_board_in_end=False, debug=False, show_delay=1):
    """ 
    This function runs one instance of a game. 
    Takes in the game, runs until the game is over. 
    Uses findings to update actor and critic. 
    
    """
    init_state = game.get_state()
    state_reward_sequence = [(init_state,0)]
    state_action_sequence = []

    while not game.is_done():

        if show_board_every_action:
            game.show_board(debug=debug, pause=show_delay)
        old_state = game.get_state()

        action = actor.get_action(game)
        if show_board_every_action:
            game.show_board_next_action(debug=debug, pause=show_delay, action=action)
        game.perform_action(action)
        if show_board_every_action:
            game.show_board_after_action(debug=debug, pause=show_delay, action=action)
        state_action_sequence.append((old_state,action))



        new_state = game.get_state()
        reward = game.get_reward()
        state_reward_sequence.append((new_state,reward))
        

        if evolve:
            delta = critic.calculate_td_error(old_state, new_state, reward)
        
            critic.update(state_reward_sequence)
            actor.update(delta, state_action_sequence)

    if show_board_in_end:
        game.show_board(debug=debug, pause=show_delay)     

def run_ai(board_type, board_size, times, actor, critic, empty_nodes_pos=[],evolve=True,show_board_every_action=False, show_board_in_end=False,debug=False, show_delay=1):
    """
    Runs several games in a row. 
    Updates actor and critic. 
    Returns a list with the number of remaining pegs for every game that is played.
    """
    results = []
    for i in range(times):
        progress_bar(i+1,times)
        game = PegSolitaire(board_type,board_size, empty_nodes_pos)
        run_one_game(game,a,c,evolve,show_board_every_action, show_board_in_end,debug,show_delay)
        results.append(game.get_end_result())
    return results

def print_averages(results, average_amount):
    """
    Uses pyplot to show the results. 
    results: List of remaining pegs for each game
    average_amount: Amount of result elements that should be taken the average. 
    """

    plt.close()
    xs = []
    ys = []

    temp_amount = 0
    temp_times = 0
    for i in range(len(results)):
        temp_amount += results[i]
        temp_times += 1
        if i % average_amount == 0:
            xs.append(i)
            ys.append(temp_amount/temp_times)
            temp_amount = 0
            temp_times = 0
    plt.xlabel("Epochs")
    plt.ylabel("Average remaining pegs")
    plt.plot(xs,ys)
    plt.show()

def progress_bar(current_step, total_steps, print_interval=20,update_all_steps=True):
    """ Shows a progress bar that updates inline and keeps track of the progress state"""

    if print_interval > total_steps:
        print_interval = total_steps
    hashtags = current_step*print_interval//total_steps
    if update_all_steps or current_step%(total_steps//print_interval) == 0:
        print(f"\rStatus: [{'#'*(hashtags)}{'.'*(print_interval-hashtags)}] {current_step}/{total_steps}", end="", flush=True)
    if current_step==total_steps:
        print(f"\rStatus: Done" + 60*" ")

def state_size(board_type, board_size):
    """ Calculates the amount of nodes in the game / state size. """
    if board_type == 'Diamond':
        return board_size**2
    elif board_type == 'Triangle':
        num = board_size
        while board_size > 0:
            board_size -= 1
            num += board_size
        return num

def create_critic(lam,alpha,gamma,table,state_size=None,layers=None, adapt_end_layers=True, update_whole_sequence=False):
    """ Creates a critic based on the arguments given """
    if table:
        return TableCritic(lam,alpha,gamma)
    else: 
        if ADAPT_END_LAYERS:
            adapted_layers = [state_size] + LAYERS[1:-1] + [1]
            return ANNCritic(lam,alpha,gamma,adapted_layers,update_whole_sequence)
        return ANNCritic(lam,alpha,gamma,LAYERS,update_whole_sequence)


if __name__ == '__main__':

    state_size = state_size(board_type, board_size)

    a = Actor(lam,alpha_actor,gamma,epsilon)
    c = create_critic(lam,alpha_critic,gamma,USE_TABLE,state_size,LAYERS, ADAPT_END_LAYERS, UPDATE_WHOLE_SEQUENCE)

    if SCENARIO in SCENARIO_DESCRIPTIONS:
        print(f'\nScenario: {SCENARIO}\nDescription: {SCENARIO_DESCRIPTIONS[SCENARIO]}\n')
    else: 
        print(f'\nUndefined scenario\n\nRunning scenario: {0}\nDescription: {SCENARIO_DESCRIPTIONS[0]}\n')

    print('Training')
    results = run_ai(board_type,board_size,times_with_learning,a,c,empty_nodes_pos)
    a.epsilon = 0

    if times_after_learning_testing:
        print('\nTesting policy with greedy behaviour')
        results += run_ai(board_type,board_size,times_after_learning_testing,a,c,empty_nodes_pos,False,False,False)

    if times_after_learning_with_show_last_move:
        print('\nTesting policy and showing only last move')
        results += run_ai(board_type,board_size,times_after_learning_with_show_last_move,a,c,empty_nodes_pos,False,False,True,debug,delay)

    if times_after_learning_with_show_every_move:
        print('\nTesting policy and showing every move')
        results += run_ai(board_type,board_size,times_after_learning_with_show_every_move,a,c,empty_nodes_pos,False,True,True,debug,delay)

    if show_statistics:
        print('\nShowing statistics')
        print_averages(results,average_amount)

