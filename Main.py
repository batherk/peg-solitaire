from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
import matplotlib.pyplot as plt

# Parameter settings
lam = 0.5
alpha = 0.001
gamma = 0.9
epsilon = 0.15

# Game settings
board_type = "Triangle"
board_size = 5
empty_nodes_pos = []

# Epoch settings
times_with_learning = 3000
times_after_learning_testing = 1000
times_after_learning_with_show_every_move = 0
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
USE_TABLE = True
HIDDEN_LAYERS = [10,5]

def run_one_game(game,actor,critic,evolve=True,show_board_every_action=False, show_board_in_end=False, debug=False, show_delay=1):

    init_state = game.get_state()
    state_sequence = [init_state]
    state_action_sequence = []

    while not game.is_done():
        if show_board_every_action:
            game.show_board(debug=debug, pause=show_delay)
        old_state = game.get_state()

        action = actor.get_action(game)
        game.perform_action(action)
        state_action_sequence.append((old_state,action))

        new_state = game.get_state()
        state_sequence.append(new_state)
        reward = game.get_reward()

        if evolve:
            delta = critic.calculate_td_error(old_state, new_state, reward)
        
            critic.update(state_sequence)
            actor.update(delta, state_action_sequence)
    if show_board_in_end:
        game.show_board(debug=debug, pause=show_delay)     

def run_ai(board_type, board_size, times, actor, critic, empty_nodes_pos=[],evolve=True,show_board_every_action=False, show_board_in_end=False,debug=False, show_delay=1):
    results = []
    for i in range(times):
        progress_bar(i+1,times)
        game = PegSolitaire(board_type,board_size, empty_nodes_pos)
        run_one_game(game,a,c,evolve,show_board_every_action, show_board_in_end,debug,show_delay)
        results.append(game.get_end_result())
    return results

def print_averages(results, average_amount):
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
    plt.plot(xs,ys)
    plt.show()

def progress_bar(current_step, total_steps, print_interval=20,update_all_steps=True):
    if print_interval > total_steps:
        print_interval = total_steps
    hashtags = current_step*print_interval//total_steps
    if update_all_steps or current_step%(total_steps//print_interval) == 0:
        print(f"\rStatus: [{'#'*(hashtags)}{'.'*(print_interval-hashtags)}] {current_step}/{total_steps}", end="", flush=True)
    if current_step==total_steps:
        print(f"\rStatus: Done" + 60*" ")

def state_size(board_type, board_size):
    if board_type == 'Diamond':
        return board_size**2
    elif board_type == 'Triangle':
        num = board_size
        while board_size > 0:
            board_size -= 1
            num += board_size
        return num

def create_critic(lam,alpha,gamma,table,state_size=None,hidden_layers=None):
    if table:
        return TableCritic(lam,alpha,gamma)
    else: 
        layers = [state_size] + hidden_layers + [1]
        return ANNCritic(lam,alpha,gamma,layers)

if __name__ == '__main__':

    state_size = state_size(board_type, board_size)

    a = Actor(lam,alpha,gamma,epsilon)
    c = create_critic(lam,alpha,gamma,USE_TABLE,state_size,HIDDEN_LAYERS)

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


