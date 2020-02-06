
class Critic:
    """ Evaluates the states.  Maps state to value of expected future reward """
    
    def __init__(self, lam, alpha, gamma):
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma

class TableCritic(Critic):
    """ Evaluates the states.  Maps state to value of expected future reward using a dictionary """

    def __init__(self, lam, alpha, gamma):
        super(TableCritic, self).__init__(lam,alpha, gamma)
        self.expected = {}
        self.trace = {}

    def calculate_td_error(self, old_state, new_state, reward):
        if new_state not in self.expected:
            self.expected[new_state] = 0
        return reward + self.gamma*self.expected[new_state] - self.expected[old_state]

    def update(self, delta, sequence):
        self.trace[sequence[-1]] = 1
        for state in sequence:
            self.expected[state] += self.alpha * delta * self.trace[state]
            self.trace[state] *= self.gamma * self.lam

    def add_init_state_to_tables(self, state, actions):
        if not state in self.expected:
            self.expected[state] = 0
        if not state in self.trace:
            self.trace[state]= 0



class ANNCritic(Critic):
    """ Evaluates the states.  Maps state to value of expected future reward using an artificial neural network """

    def __init__(self):
        super(ANNCritic, self).__init__()

