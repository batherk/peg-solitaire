from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
import matplotlib.pyplot as plt


# Scenario: 
SCENARIO = 0

SCENARIO_DESCRIPTIONS = {}
SCENARIO_DESCRIPTIONS[0] = "Custom settings"
SCENARIO_DESCRIPTIONS[1] = "Triangle, 5, Table, (0,0) is empty, 500 epochs, ALPHA_ACTOR: 0.001, ALPHA_CRITIC: 0.001."
SCENARIO_DESCRIPTIONS[2] = "Diamond, 4, Table, (1,2) is empty, 1000 epochs, ALPHA_ACTOR: 0.001, ALPHA_CRITIC: 0.001." 
SCENARIO_DESCRIPTIONS[3] = "Triangle, 5, Neural Net, (0,0) is empty, 500 epochs, ALPHA_ACTOR: 0.001, ALPHA_CRITIC: 0.0001."
SCENARIO_DESCRIPTIONS[4] = "Diamond, 4, Neural Net, (1,2) is empty, 1000 epochs, ALPHA_ACTOR: 0.001, ALPHA_CRITIC: 0.0001."

# Parameter settings
LAMBDA = 0.5                    # Trace decay factor
GAMMA = 0.9                     # Discount factor
EPSILON = 0.15                  # Greediness factor

ALPHA_CRITIC = 0.001            # Learning rate critic
ALPHA_ACTOR = 0.001             # Learning rate actor

# Game settings
BOARD_TYPE = "Triangle"         # Board can be "Triangle" or "Diamond"
BOARD_SIZE = 5              
EMPTY_NODES_POS = []            # If this is empty the game initializes with a random peg every time, else you set the empty nodes

# Epoch settings
EPOCHS = 3000

# Checking policy after epsilon is set to 0 -> Greedy Policy
ITERATIONS_TEST = 500               # Number of times the policy wil be tested
ITERATIONS_SHOW_ALL_ACTIONS = 1     # Number of times the policy's handling of a game will be shown
ITERATIONS_SHOW_END_STATE = 0       # Number of times the end state of the game will be shown

# Statistics settings
SHOW_STATISTICS = True          # If you want to see a graph with the policy's performance
SHOW_REMAINING_PEGS = True      # If you want to see the graph with remaining pegs
SHOW_WINS = False               # If you want to see the amount of times the policy wins the game
GRAPH_POINTS = 100          
AVERAGE_AMOUNT = (EPOCHS + ITERATIONS_TEST + ITERATIONS_SHOW_ALL_ACTIONS + ITERATIONS_SHOW_END_STATE)//GRAPH_POINTS

# Plot settings
DELAY_END_STATE = 0.1           # Delay between each showing of end states
DELAY_EVERY_ACTION = 0.2        # Delay between each showning of action and consequence

# Debug settings
DEBUG = False

# Critic settings
USE_TABLE = True                # True -> Table, False -> Neural net

# Neural net critic
UPDATE_WHOLE_SEQUENCE = True    # If true the net is updated by every state in the sequence, based on eligibility traces. Only one time per state per episode. 
LAYERS = [15,10,10,10,1]        # This works: LAYERS = [X,10,10,10,X], ADAPT_END_LAYERS = True
ADAPT_END_LAYERS = True         # If true the first layer of nodes gets the dimension of the state size of the game and last layer gets one node, else custom


# Setting constants for the different scenarios
if SCENARIO == 1:
    USE_TABLE = True
    BOARD_TYPE = "Triangle"
    BOARD_SIZE = 5
    EPOCHS = 500
    ALPHA_CRITIC = 0.001
    ALPHA_ACTOR = 0.001
    EMPTY_NODES_POS = [(0,0)]
elif SCENARIO == 2:
    USE_TABLE = True
    BOARD_TYPE = "Diamond"
    BOARD_SIZE = 4
    EPOCHS = 1000
    ALPHA_CRITIC = 0.001
    ALPHA_ACTOR = 0.001
    EMPTY_NODES_POS = [(1,2)]
elif SCENARIO == 3:
    USE_TABLE = False
    BOARD_TYPE = "Triangle"
    BOARD_SIZE = 5
    EPOCHS = 500
    ALPHA_CRITIC = 0.0001
    ALPHA_ACTOR = 0.001
    EMPTY_NODES_POS = [(0,0)]
elif SCENARIO == 4:
    USE_TABLE = False
    BOARD_TYPE = "Diamond"
    BOARD_SIZE = 4
    EPOCHS = 1000
    ALPHA_CRITIC = 0.00005
    ALPHA_ACTOR = 0.001
    EMPTY_NODES_POS = [(1,2)]



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

def print_averages(results, average_amount, show_remaining_pegs=True, show_wins=True):
    """
    Uses pyplot to show the results. 
    results: List of remaining pegs for each game
    average_amount: Amount of result elements that should be taken the average. 
    """

    plt.close()

    fig, ax1 = plt.subplots()

    xs = []
    remaining_pegs = []
    wins = []
    
    temp_wins = 0
    temp_remaining_pegs = 0
    temp_times = 0

    for i in range(len(results)):
        temp_remaining_pegs += results[i]
        temp_times += 1

        if results[i] == 1:
            temp_wins += 1

        if i % average_amount == 0:
            xs.append(i)
            remaining_pegs.append(temp_remaining_pegs/temp_times)
            wins.append(temp_wins)

            temp_remaining_pegs = 0
            temp_times = 0
            temp_wins = 0

    
    color = 'tab:blue'
    if show_remaining_pegs: 
        ax1.plot(xs, remaining_pegs, color=color)
        ax1.set_ylabel('Average remaining pegs', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()

    color = 'tab:green'
    if show_wins: 
        ax2.plot(xs, wins, color=color)
        ax2.set_ylabel(f'Amount of wins per {average_amount} games', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xlabel('Epochs')
    fig.tight_layout()
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

    state_size = state_size(BOARD_TYPE, BOARD_SIZE)

    a = Actor(LAMBDA,ALPHA_ACTOR,GAMMA,EPSILON)
    c = create_critic(LAMBDA,ALPHA_CRITIC,GAMMA,USE_TABLE,state_size,LAYERS, ADAPT_END_LAYERS, UPDATE_WHOLE_SEQUENCE)

    if SCENARIO in SCENARIO_DESCRIPTIONS:
        print(f'\nScenario: {SCENARIO}\nDescription: {SCENARIO_DESCRIPTIONS[SCENARIO]}\n')
    else: 
        print(f'\nUndefined scenario\n\nRunning scenario: {0}\nDescription: {SCENARIO_DESCRIPTIONS[0]}\n')

    print('Training')
    results = run_ai(BOARD_TYPE,BOARD_SIZE,EPOCHS,a,c,EMPTY_NODES_POS)
    a.epsilon = 0

    if ITERATIONS_TEST:
        print('\nTesting policy with greedy behaviour')
        results += run_ai(BOARD_TYPE,BOARD_SIZE,ITERATIONS_TEST,a,c,EMPTY_NODES_POS,False,False,False)

    if ITERATIONS_SHOW_END_STATE:
        print('\nTesting policy and showing only last move')
        results += run_ai(BOARD_TYPE,BOARD_SIZE,ITERATIONS_SHOW_END_STATE,a,c,EMPTY_NODES_POS,False,False,True,DEBUG,DELAY_END_STATE)

    if ITERATIONS_SHOW_ALL_ACTIONS:
        print('\nTesting policy and showing every move')
        results += run_ai(BOARD_TYPE,BOARD_SIZE,ITERATIONS_SHOW_ALL_ACTIONS,a,c,EMPTY_NODES_POS,False,True,True,DEBUG,DELAY_EVERY_ACTION)

    if SHOW_STATISTICS:
        print('\nShowing statistics')
        print_averages(results,AVERAGE_AMOUNT,SHOW_REMAINING_PEGS, SHOW_WINS)

