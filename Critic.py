#from tensorflow import keras

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
        for state in [old_state, new_state]:
            if state not in self.expected:
                self.expected[state] = 0
        return reward + self.gamma*self.expected[new_state] - self.expected[old_state]

    def update(self, delta, sequence):
        if len(sequence)==2:
            self.trace[sequence[0]] = self.gamma * self.lam
        self.trace[sequence[-1]] = 1
        for state in sequence:
            self.expected[state] += self.alpha * delta * self.trace[state]
            self.trace[state] *= self.gamma * self.lam




class ANNCritic(Critic):
    """ Evaluates the states.  Maps state to value of expected future reward using an artificial neural network """

    def __init__(self, lam, alpha, gamma):
        super(TableCritic, self).__init__(lam,alpha, gamma)
        #self.model = keras.Sequential([
        #    keras.layers.Dense(10, activation="relu")
        #])

