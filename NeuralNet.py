import torch

criterion = torch.nn.MSELoss()

class NeuralNet(torch.nn.Module):

    def __init__(self,layer_nodes):
        super(NeuralNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.criterion = torch.nn.MSELoss()
        for i in range(len(layer_nodes)-1):
            self.layers.append(torch.nn.Linear(layer_nodes[i], layer_nodes[i+1]))
            
    
    def forward(self,x):
        """ Overwrites the forward function of the pytorch neural net, so ouput is correctly given"""
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x))
        return x

    def loss(self, output, target):
        """ Computes the loss. Is used to update the net"""
        return self.criterion(output, target)

    def update_weights(self, loss, learning_rate, eligibility_trace=1,retain_graph=False):
        """ Update the weights of the neural net based on the inputs """
        self.zero_grad()
        loss.backward(retain_graph=retain_graph)
        for f in self.parameters():
            f.data.sub_(f.grad.data * learning_rate * eligibility_trace)
