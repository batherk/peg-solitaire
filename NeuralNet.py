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
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x))
        return x

    def loss(self, output, target):
        return self.criterion(output, target)

    def update_weights(self, loss, alpha, eligibility_trace=1):
        self.zero_grad()
        loss.backward()
        for f in self.parameters():
            f.data.sub_(f.grad.data * alpha * eligibility_trace)


    def get_weights(self,layer):
        return self.layers[layer].weight
