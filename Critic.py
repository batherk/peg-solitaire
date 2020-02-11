from NeuralNet import NeuralNet, torch

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
        self.delta = None

    def calculate_td_error(self, old_state, new_state, reward):
        for state in [old_state, new_state]:
            if state not in self.expected:
                self.expected[state] = 0
        self.delta = reward + self.gamma*self.expected[new_state] - self.expected[old_state]
        return self.delta

    def update(self, sequence):
        if len(sequence)==2:
            self.trace[sequence[0]] = self.gamma * self.lam
        self.trace[sequence[-1]] = 1
        for state in sequence:
            self.expected[state] += self.alpha * self.delta * self.trace[state]
            self.trace[state] *= self.gamma * self.lam

class ANNCritic(Critic):
    """ Evaluates the states.  Maps state to value of expected future reward using an artificial neural network """

    def __init__(self, lam, alpha, gamma, layers):
        super(ANNCritic, self).__init__(lam,alpha, gamma)
        self.net = NeuralNet(layers)
        self.trace = {}
        self.loss = None

    def calculate_td_error(self, old_state, new_state, reward):
        output = self.net(self.state_tensor_convert(old_state))
        target = self.net(self.state_tensor_convert(new_state)) + reward
        self.loss = self.net.loss(output,target)
        return self.loss.item()

    def update(self, sequence):
        if len(sequence)==2:
            self.trace[sequence[0]] = self.gamma * self.lam
        self.trace[sequence[-1]] = 1
        for i,state in enumerate(sequence):
            if i < len(sequence)-1:
                self.net.update_weights(self.loss, self.alpha, self.trace[state],True)
                self.trace[state] *= self.gamma * self.lam

    def state_tensor_convert(self,state):
        return torch.Tensor(state)

