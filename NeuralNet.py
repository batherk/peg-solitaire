import torch

class NeuralNet(torch.nn.Module):

    def __init__(self,layer_nodes):
        super(NeuralNet, self).__init__()
        self.layers = []
        for i in range(len(layer_nodes)-1):
            self.layers.append(torch.nn.Linear(layer_nodes[i], layer_nodes[i+1]))
    
    def forward(self,x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        return x


a = NeuralNet((8,4,1))
print(a.layers)