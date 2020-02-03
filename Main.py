from PegSolitaire import PegSolitaire
from Actor import Actor
from Critic import *
import matplotlib.pyplot as plt

lam = 0.5
alpha = 0.001
gamma = 0.9
epsilon = 0.15
average_amount = 50
times_with_learning = 5000
times_after_learning = 1000
time_after_learning_with_show = 10
board_type = "Triangle"
board_size = 5
empty_nodes_pos = []

a = Actor(lam,alpha,gamma,epsilon)
c = TableCritic(lam,alpha,gamma)

def run_one_game(game,actor,critic,evolve=True,show_board_every_action=False, show_board_in_end=False):

    init_state = game.get_state()
    state_sequence = [init_state]
    state_action_sequence = []
    critic.add_init_state_to_tables(init_state,game.get_possible_actions())

    while not game.is_done():
        if show_board_every_action:
            game.show_board()
        old_state = game.get_state()

        action = actor.get_action(game)
        game.perform_action(action)
        state_action_sequence.append((old_state,action))

        new_state = game.get_state()
        state_sequence.append(new_state)
        reward = game.get_reward()

        if evolve:
            delta = critic.calculate_td_error(old_state, new_state, reward)
        
            critic.update_tables(delta, state_sequence)
            actor.update_tables(delta, state_action_sequence)
    if show_board_in_end:
        game.show_board()     

def run_ai(board_type, board_size, times, actor, critic, empty_nodes_pos=[],evolve=True,show_board_every_action=False, show_board_in_end=False):
    results = []
    for i in range(times):
        progress_bar(i+1,times)
        game = PegSolitaire(board_type,board_size, empty_nodes_pos)
        run_one_game(game,a,c,evolve,show_board_every_action, show_board_in_end)
        results.append(game.get_end_result())
    return results

def print_averages(results, average_amount):
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

def print_expected_reward_for_win_states(critic):
    for state in critic.expected:
        if state.count('1') == 1:
            print(f'State: {state} Expected future reward: {critic.expected[state]}')

def print_expected_reward_for_states(critic):
    for state in critic.expected:
        print(f'State: {state} Expected future reward: {critic.expected[state]}')

def progress_bar(current_step, total_steps, print_interval=20):
    if print_interval > total_steps:
        print_interval = total_steps
    hashtags = current_step*print_interval//total_steps
    if current_step%(total_steps//print_interval) == 0:
        print(f"\rStatus: [{'#'*(hashtags)}{'.'*(print_interval-hashtags)}] {current_step}/{total_steps}", end="", flush=True)
    if current_step==total_steps:
        print(f"\rStatus: Done" + 60*" ")



print("Training")
results_while_training = run_ai(board_type,board_size,times_with_learning,a,c,empty_nodes_pos)
a.epsilon = 0
print("\nTesting policy with greedy behaviour")
results_after_training = run_ai(board_type,board_size,times_after_learning,a,c,empty_nodes_pos,False,False,False)

all_results = results_while_training + results_after_training

print_averages(all_results,average_amount)

print("\nShowing outcomes")
results = run_ai(board_type,board_size,time_after_learning_with_show,a,c,empty_nodes_pos,False,False,True)





